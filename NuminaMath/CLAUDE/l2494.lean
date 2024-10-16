import Mathlib

namespace NUMINAMATH_CALUDE_factorization_equality_l2494_249458

theorem factorization_equality (m a : ℝ) : m * a^2 - m = m * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2494_249458


namespace NUMINAMATH_CALUDE_horizontal_asymptote_of_f_l2494_249438

noncomputable def f (x : ℝ) : ℝ := (8 * x^2 - 4) / (4 * x^2 + 8 * x + 3)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N, ∀ x > N, |f x - 2| < ε :=
sorry

end NUMINAMATH_CALUDE_horizontal_asymptote_of_f_l2494_249438


namespace NUMINAMATH_CALUDE_correct_contribution_l2494_249406

/-- The cost of the project in billions of dollars -/
def project_cost : ℝ := 25

/-- The number of participants in millions -/
def num_participants : ℝ := 300

/-- The contribution required from each participant -/
def individual_contribution : ℝ := 83

/-- Theorem stating that the individual contribution is correct given the project cost and number of participants -/
theorem correct_contribution : 
  (project_cost * 1000) / num_participants = individual_contribution := by
  sorry

end NUMINAMATH_CALUDE_correct_contribution_l2494_249406


namespace NUMINAMATH_CALUDE_semi_circle_radius_equals_rectangle_area_l2494_249479

theorem semi_circle_radius_equals_rectangle_area (length width : ℝ) (h1 : length = 8) (h2 : width = Real.pi) :
  ∃ (r : ℝ), r = 4 ∧ (1/2 * Real.pi * r^2) = (length * width) :=
by sorry

end NUMINAMATH_CALUDE_semi_circle_radius_equals_rectangle_area_l2494_249479


namespace NUMINAMATH_CALUDE_volume_of_rotated_region_l2494_249467

/-- The volume of a solid formed by rotating a region consisting of a 6x1 rectangle
    and a 4x3 rectangle about the x-axis -/
theorem volume_of_rotated_region : ℝ := by
  -- Define the dimensions of the rectangles
  let height1 : ℝ := 6
  let width1 : ℝ := 1
  let height2 : ℝ := 3
  let width2 : ℝ := 4

  -- Define the volumes of the two cylinders
  let volume1 : ℝ := Real.pi * height1^2 * width1
  let volume2 : ℝ := Real.pi * height2^2 * width2

  -- Define the total volume
  let total_volume : ℝ := volume1 + volume2

  -- Prove that the total volume equals 72π
  have : total_volume = 72 * Real.pi := by sorry

  -- Return the result
  exact 72 * Real.pi

end NUMINAMATH_CALUDE_volume_of_rotated_region_l2494_249467


namespace NUMINAMATH_CALUDE_evaluate_fraction_l2494_249405

theorem evaluate_fraction (a b : ℤ) (h1 : a = 7) (h2 : b = -3) :
  3 / (a - b) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_fraction_l2494_249405


namespace NUMINAMATH_CALUDE_min_sum_of_coefficients_l2494_249457

theorem min_sum_of_coefficients (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x y : ℝ, a * x + b * y + 1 = 0 ∧ x^2 + y^2 + 8*x + 2*y + 1 = 0) →
  a + b ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_coefficients_l2494_249457


namespace NUMINAMATH_CALUDE_cone_volume_l2494_249499

/-- The volume of a cone given specific conditions -/
theorem cone_volume 
  (p q : ℝ) 
  (a : ℝ) 
  (α : ℝ) 
  (h_p : p > 0) 
  (h_q : q > 0) 
  (h_a : a > 0) 
  (h_α : 0 < α ∧ α < π / 2) :
  let V := π * a^3 / (3 * Real.sin α * Real.cos α^2 * Real.cos (π * q / (p + q))^2)
  ∃ (r h : ℝ), r > 0 ∧ h > 0 ∧ V = (1/3) * π * r^2 * h :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l2494_249499


namespace NUMINAMATH_CALUDE_repetend_of_four_seventeenths_l2494_249493

/-- The decimal representation of 4/17 has a 6-digit repetend of 235294 -/
theorem repetend_of_four_seventeenths : ∃ (a b : ℕ), 
  (4 : ℚ) / 17 = (a : ℚ) / 999999 + (b : ℚ) / (999999 * 1000000) ∧ 
  a = 235294 ∧ 
  b < 999999 := by sorry

end NUMINAMATH_CALUDE_repetend_of_four_seventeenths_l2494_249493


namespace NUMINAMATH_CALUDE_dan_total_limes_l2494_249476

/-- The number of limes Dan picked -/
def limes_picked : ℕ := 9

/-- The number of limes Sara gave to Dan -/
def limes_given : ℕ := 4

/-- The total number of limes Dan has now -/
def total_limes : ℕ := limes_picked + limes_given

theorem dan_total_limes : total_limes = 13 := by
  sorry

end NUMINAMATH_CALUDE_dan_total_limes_l2494_249476


namespace NUMINAMATH_CALUDE_parity_condition_l2494_249420

theorem parity_condition (n : ℕ) : n ≥ 2 →
  (∀ i j : ℕ, 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n →
    Even (i + j) ↔ Even (Nat.choose n i + Nat.choose n j)) ↔
  ∃ k : ℕ, k ≥ 1 ∧ n = 2^k - 2 :=
by sorry

end NUMINAMATH_CALUDE_parity_condition_l2494_249420


namespace NUMINAMATH_CALUDE_unoccupied_area_formula_l2494_249492

/-- The area of a rectangle not occupied by a hole and a square -/
def unoccupied_area (x : ℝ) : ℝ :=
  let large_rect := (2*x + 9) * (x + 6)
  let hole := (x - 1) * (2*x - 5)
  let square := (x + 3)^2
  large_rect - hole - square

/-- Theorem stating the unoccupied area in terms of x -/
theorem unoccupied_area_formula (x : ℝ) :
  unoccupied_area x = -x^2 + 22*x + 40 := by
  sorry

end NUMINAMATH_CALUDE_unoccupied_area_formula_l2494_249492


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2494_249454

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2494_249454


namespace NUMINAMATH_CALUDE_weeks_to_work_is_ten_l2494_249489

/-- The number of weeks Isabelle must work to afford concert tickets for herself and her brothers -/
def weeks_to_work : ℕ :=
let isabelle_ticket_cost : ℕ := 20
let brother_ticket_cost : ℕ := 10
let number_of_brothers : ℕ := 2
let total_savings : ℕ := 10
let weekly_earnings : ℕ := 3
let total_ticket_cost : ℕ := isabelle_ticket_cost + brother_ticket_cost * number_of_brothers
let additional_money_needed : ℕ := total_ticket_cost - total_savings
(additional_money_needed + weekly_earnings - 1) / weekly_earnings

theorem weeks_to_work_is_ten : weeks_to_work = 10 := by
  sorry

end NUMINAMATH_CALUDE_weeks_to_work_is_ten_l2494_249489


namespace NUMINAMATH_CALUDE_simplify_expression_l2494_249432

theorem simplify_expression (s : ℝ) : 120 * s - 32 * s = 88 * s := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2494_249432


namespace NUMINAMATH_CALUDE_eighth_term_is_21_l2494_249481

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem eighth_term_is_21 :
  fibonacci 7 = 21 ∧ fibonacci 8 = 34 ∧ fibonacci 9 = 55 :=
by sorry

end NUMINAMATH_CALUDE_eighth_term_is_21_l2494_249481


namespace NUMINAMATH_CALUDE_melanie_coin_count_l2494_249444

/-- Represents the number of coins Melanie has or receives -/
structure CoinCount where
  dimes : ℕ
  nickels : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in dollars -/
def coinValue (coins : CoinCount) : ℚ :=
  (coins.dimes * 10 + coins.nickels * 5 + coins.quarters * 25) / 100

/-- Adds two CoinCount structures -/
def addCoins (a b : CoinCount) : CoinCount :=
  { dimes := a.dimes + b.dimes,
    nickels := a.nickels + b.nickels,
    quarters := a.quarters + b.quarters }

def initial : CoinCount := { dimes := 19, nickels := 12, quarters := 8 }
def fromDad : CoinCount := { dimes := 39, nickels := 22, quarters := 15 }
def fromSister : CoinCount := { dimes := 15, nickels := 7, quarters := 12 }
def fromMother : CoinCount := { dimes := 25, nickels := 10, quarters := 0 }
def fromGrandmother : CoinCount := { dimes := 0, nickels := 30, quarters := 3 }

theorem melanie_coin_count :
  let final := addCoins initial (addCoins fromDad (addCoins fromSister (addCoins fromMother fromGrandmother)))
  final.dimes = 98 ∧
  final.nickels = 81 ∧
  final.quarters = 38 ∧
  coinValue final = 2335 / 100 := by
  sorry

end NUMINAMATH_CALUDE_melanie_coin_count_l2494_249444


namespace NUMINAMATH_CALUDE_centers_connection_line_l2494_249465

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem centers_connection_line :
  ∃ (x1 y1 x2 y2 : ℝ),
    (∀ x y, circle1 x y ↔ (x - x1)^2 + (y - y1)^2 = (x1^2 + y1^2)) ∧
    (∀ x y, circle2 x y ↔ (x - x2)^2 + (y - y2)^2 = x2^2) ∧
    line_equation x1 y1 ∧
    line_equation x2 y2 :=
sorry

end NUMINAMATH_CALUDE_centers_connection_line_l2494_249465


namespace NUMINAMATH_CALUDE_remainder_problem_l2494_249433

theorem remainder_problem (N : ℤ) (h : N % 242 = 100) : N % 18 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2494_249433


namespace NUMINAMATH_CALUDE_train_speed_l2494_249441

/-- The speed of a train crossing a platform of equal length -/
theorem train_speed (train_length platform_length : ℝ) (crossing_time : ℝ) : 
  train_length = platform_length → 
  train_length = 600 → 
  crossing_time = 1 / 60 → 
  (train_length + platform_length) / crossing_time / 1000 = 72 :=
by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2494_249441


namespace NUMINAMATH_CALUDE_locus_of_center_l2494_249447

-- Define the circle C
def circle_C (a x y : ℝ) : Prop :=
  x^2 + y^2 - (2*a^2 - 4)*x - 4*a^2*y + 5*a^4 - 4 = 0

-- Define the locus equation
def locus_equation (x y : ℝ) : Prop :=
  2*x - y + 4 = 0

-- Define the x-coordinate range
def x_range (x : ℝ) : Prop :=
  -2 ≤ x ∧ x < 0

-- Theorem statement
theorem locus_of_center :
  ∀ a x y : ℝ, circle_C a x y → 
  ∃ h k : ℝ, (locus_equation h k ∧ x_range h) ∧
  (∀ x' y' : ℝ, locus_equation x' y' ∧ x_range x' → 
   ∃ a' : ℝ, circle_C a' x' y') :=
sorry

end NUMINAMATH_CALUDE_locus_of_center_l2494_249447


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2494_249437

theorem geometric_series_sum : 
  let a : ℝ := (Real.sqrt 2 + 1) / (Real.sqrt 2 - 1)
  let q : ℝ := (Real.sqrt 2 - 1) / Real.sqrt 2
  let S : ℝ := a / (1 - q)
  S = 6 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2494_249437


namespace NUMINAMATH_CALUDE_bananas_left_l2494_249471

/-- The number of bananas in a dozen -/
def dozen : ℕ := 12

/-- The number of bananas Elizabeth ate -/
def eaten : ℕ := 4

/-- Theorem: If Elizabeth bought a dozen bananas and ate 4 of them, then 8 bananas are left -/
theorem bananas_left (bought : ℕ) (ate : ℕ) (h1 : bought = dozen) (h2 : ate = eaten) :
  bought - ate = 8 := by
  sorry

end NUMINAMATH_CALUDE_bananas_left_l2494_249471


namespace NUMINAMATH_CALUDE_acetone_weight_approx_l2494_249464

/-- Atomic weight of Carbon in amu -/
def carbon_weight : Float := 12.01

/-- Atomic weight of Hydrogen in amu -/
def hydrogen_weight : Float := 1.008

/-- Atomic weight of Oxygen in amu -/
def oxygen_weight : Float := 16.00

/-- Number of Carbon atoms in Acetone -/
def carbon_count : Nat := 3

/-- Number of Hydrogen atoms in Acetone -/
def hydrogen_count : Nat := 6

/-- Number of Oxygen atoms in Acetone -/
def oxygen_count : Nat := 1

/-- Calculates the molecular weight of Acetone -/
def acetone_molecular_weight : Float :=
  carbon_weight * carbon_count.toFloat +
  hydrogen_weight * hydrogen_count.toFloat +
  oxygen_weight * oxygen_count.toFloat

/-- Theorem stating that the molecular weight of Acetone is approximately 58.08 amu -/
theorem acetone_weight_approx :
  (acetone_molecular_weight - 58.08).abs < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_acetone_weight_approx_l2494_249464


namespace NUMINAMATH_CALUDE_maximal_ratio_of_primes_l2494_249412

theorem maximal_ratio_of_primes (p q : ℕ) : 
  Prime p → Prime q → p > q → ¬(240 ∣ p^4 - q^4) → 
  (∃ (r : ℚ), r = q / p ∧ r ≤ 2/3 ∧ ∀ (s : ℚ), s = q / p → s ≤ r) :=
sorry

end NUMINAMATH_CALUDE_maximal_ratio_of_primes_l2494_249412


namespace NUMINAMATH_CALUDE_inequality_addition_l2494_249422

theorem inequality_addition (m n c : ℝ) : m > n → m + c > n + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l2494_249422


namespace NUMINAMATH_CALUDE_alloy_mixture_problem_l2494_249462

/-- Represents the composition of an alloy -/
structure Alloy where
  lead : ℝ
  tin : ℝ
  copper : ℝ

/-- The total weight of an alloy -/
def Alloy.weight (a : Alloy) : ℝ := a.lead + a.tin + a.copper

/-- The problem statement -/
theorem alloy_mixture_problem (alloyA alloyB : Alloy) 
  (h1 : alloyA.weight = 170)
  (h2 : alloyB.weight = 250)
  (h3 : alloyB.tin / alloyB.copper = 3 / 5)
  (h4 : alloyA.tin + alloyB.tin = 221.25)
  (h5 : alloyA.copper = 0)
  (h6 : alloyB.lead = 0) :
  alloyA.lead / alloyA.tin = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_alloy_mixture_problem_l2494_249462


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_l2494_249421

theorem consecutive_even_numbers (x y z : ℕ) : 
  (∃ n : ℕ, x = 2*n ∧ y = 2*n + 2 ∧ z = 2*n + 4) →  -- x, y, z are consecutive even numbers
  (x + y + z = (x + y + z) / 3 + 44) →               -- sum is 44 more than average
  z = 24                                             -- largest number is 24
:= by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_l2494_249421


namespace NUMINAMATH_CALUDE_average_licks_to_center_l2494_249468

def dan_licks : ℕ := 58
def michael_licks : ℕ := 63
def sam_licks : ℕ := 70
def david_licks : ℕ := 70
def lance_licks : ℕ := 39

def total_licks : ℕ := dan_licks + michael_licks + sam_licks + david_licks + lance_licks
def num_people : ℕ := 5

theorem average_licks_to_center (h : total_licks = dan_licks + michael_licks + sam_licks + david_licks + lance_licks) :
  (total_licks : ℚ) / num_people = 60 := by sorry

end NUMINAMATH_CALUDE_average_licks_to_center_l2494_249468


namespace NUMINAMATH_CALUDE_prism_volume_l2494_249483

/-- The volume of a right rectangular prism with face areas 10, 15, and 18 square inches is 30√3 cubic inches. -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 10) 
  (h2 : b * c = 15) 
  (h3 : c * a = 18) : 
  a * b * c = 30 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_prism_volume_l2494_249483


namespace NUMINAMATH_CALUDE_no_closed_broken_line_315_l2494_249486

/-- A closed broken line with the given properties -/
structure ClosedBrokenLine where
  segments : ℕ
  intersecting : Bool
  perpendicular : Bool
  symmetric : Bool

/-- The number of segments in our specific case -/
def n : ℕ := 315

/-- Theorem stating the impossibility of constructing the specified closed broken line -/
theorem no_closed_broken_line_315 :
  ¬ ∃ (line : ClosedBrokenLine), 
    line.segments = n ∧
    line.intersecting ∧
    line.perpendicular ∧
    line.symmetric :=
sorry


end NUMINAMATH_CALUDE_no_closed_broken_line_315_l2494_249486


namespace NUMINAMATH_CALUDE_daily_profit_at_35_unique_profit_600_no_profit_900_l2494_249472

/-- The daily profit function for a product -/
def P (x : ℝ) : ℝ := (x - 30) * (-2 * x + 140)

/-- The purchase price of the product -/
def purchase_price : ℝ := 30

/-- The lower bound of the selling price -/
def lower_bound : ℝ := 30

/-- The upper bound of the selling price -/
def upper_bound : ℝ := 55

theorem daily_profit_at_35 :
  P 35 = 350 := by sorry

theorem unique_profit_600 :
  ∃! x : ℝ, lower_bound ≤ x ∧ x ≤ upper_bound ∧ P x = 600 ∧ x = 40 := by sorry

theorem no_profit_900 :
  ¬ ∃ x : ℝ, lower_bound ≤ x ∧ x ≤ upper_bound ∧ P x = 900 := by sorry

end NUMINAMATH_CALUDE_daily_profit_at_35_unique_profit_600_no_profit_900_l2494_249472


namespace NUMINAMATH_CALUDE_circle_pi_value_l2494_249446

theorem circle_pi_value (d c : ℝ) (hd : d = 8) (hc : c = 25.12) :
  c / d = 3.14 := by sorry

end NUMINAMATH_CALUDE_circle_pi_value_l2494_249446


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_sum_l2494_249423

theorem quadratic_root_ratio_sum (x₁ x₂ : ℝ) : 
  x₁^2 - x₁ - 1 = 0 → 
  x₂^2 - x₂ - 1 = 0 → 
  x₁ ≠ 0 → 
  x₂ ≠ 0 → 
  (x₂ / x₁) + (x₁ / x₂) = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_sum_l2494_249423


namespace NUMINAMATH_CALUDE_unique_solution_l2494_249424

-- Define Θ as a natural number
variable (Θ : ℕ)

-- Define the condition that Θ is a single digit
def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

-- Define the two-digit number 4Θ
def four_Θ (Θ : ℕ) : ℕ := 40 + Θ

-- State the theorem
theorem unique_solution :
  (630 / Θ = four_Θ Θ + 2 * Θ) ∧ 
  (is_single_digit Θ) →
  Θ = 9 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2494_249424


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2494_249400

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ - 3| = 15 ∧ |x₂ - 3| = 15 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 30 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2494_249400


namespace NUMINAMATH_CALUDE_courtyard_paving_l2494_249425

-- Define the courtyard dimensions in centimeters
def courtyard_length : ℕ := 2500
def courtyard_width : ℕ := 1800

-- Define the brick dimensions in centimeters
def brick_length : ℕ := 20
def brick_width : ℕ := 10

-- Define the function to calculate the number of bricks required
def bricks_required (cl cw bl bw : ℕ) : ℕ :=
  (cl * cw) / (bl * bw)

-- Theorem statement
theorem courtyard_paving :
  bricks_required courtyard_length courtyard_width brick_length brick_width = 22500 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_paving_l2494_249425


namespace NUMINAMATH_CALUDE_antons_winning_numbers_infinite_l2494_249429

theorem antons_winning_numbers_infinite :
  ∃ (f : ℕ → ℕ), Function.Injective f ∧
  ∀ (x : ℕ), 
    let n := f x
    (¬ ∃ (m : ℕ), n = m ^ 2) ∧ 
    ∃ (k : ℕ), (n + (n+1) + (n+2) + (n+3) + (n+4)) = k ^ 2 :=
sorry

end NUMINAMATH_CALUDE_antons_winning_numbers_infinite_l2494_249429


namespace NUMINAMATH_CALUDE_fourth_score_for_average_l2494_249484

/-- Given three exam scores, this theorem proves the required fourth score to achieve a specific average. -/
theorem fourth_score_for_average (score1 score2 score3 target_avg : ℕ) :
  score1 = 87 →
  score2 = 83 →
  score3 = 88 →
  target_avg = 89 →
  ∃ (score4 : ℕ), (score1 + score2 + score3 + score4) / 4 = target_avg ∧ score4 = 98 := by
  sorry

#check fourth_score_for_average

end NUMINAMATH_CALUDE_fourth_score_for_average_l2494_249484


namespace NUMINAMATH_CALUDE_extreme_value_derivative_l2494_249409

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define what it means for f to have an extreme value at x₀
def has_extreme_value (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x₀ ≤ f x ∨ f x₀ ≥ f x

-- State the theorem
theorem extreme_value_derivative (x₀ : ℝ) :
  (has_extreme_value f x₀ → deriv f x₀ = 0) ∧
  ¬(deriv f x₀ = 0 → has_extreme_value f x₀) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_derivative_l2494_249409


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l2494_249435

-- Define the plane
variable (Plane : Type)

-- Define points
variable (O A B P P1 P2 A' B' : Plane)

-- Define the angle
variable (angle : Plane → Plane → Plane → Prop)

-- Define the property of being inside an angle
variable (insideAngle : Plane → Plane → Plane → Plane → Prop)

-- Define the property of being on a line
variable (onLine : Plane → Plane → Plane → Prop)

-- Define symmetry with respect to a line
variable (symmetricToLine : Plane → Plane → Plane → Plane → Prop)

-- Define intersection of two lines
variable (intersect : Plane → Plane → Plane → Plane → Plane → Prop)

-- Define perimeter of a triangle
variable (perimeter : Plane → Plane → Plane → ℝ)

-- Define the theorem
theorem min_perimeter_triangle
  (h1 : angle O A B)
  (h2 : insideAngle O A B P)
  (h3 : onLine O A A)
  (h4 : onLine O B B)
  (h5 : symmetricToLine P P1 O A)
  (h6 : symmetricToLine P P2 O B)
  (h7 : intersect P1 P2 O A A')
  (h8 : intersect P1 P2 O B B') :
  ∀ X Y, onLine O A X → onLine O B Y →
    perimeter P X Y ≥ perimeter P A' B' :=
  sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l2494_249435


namespace NUMINAMATH_CALUDE_train_speed_l2494_249460

/-- Proves that a train with given specifications travels at 45 km/hr -/
theorem train_speed (train_length : Real) (crossing_time : Real) (total_length : Real) :
  train_length = 130 ∧ 
  crossing_time = 30 ∧ 
  total_length = 245 → 
  (total_length - train_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2494_249460


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l2494_249485

theorem square_minus_product_plus_square : 7^2 - 4*5 + 2^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l2494_249485


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2494_249401

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + a 7 = -8)
  (h_a2 : a 2 = 2) :
  ∃ d : ℝ, d = -3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2494_249401


namespace NUMINAMATH_CALUDE_smallest_a_value_l2494_249461

/-- Represents a parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem stating the smallest possible value of a for the given parabola -/
theorem smallest_a_value (p : Parabola) 
  (vertex_x : p.a * (3/4)^2 + p.b * (3/4) + p.c = -25/16) 
  (vertex_y : -p.b / (2 * p.a) = 3/4)
  (a_positive : p.a > 0)
  (sum_integer : ∃ n : ℤ, p.a + p.b + p.c = n) :
  9 ≤ p.a ∧ ∀ a' : ℝ, 0 < a' ∧ a' < 9 → 
    ¬∃ (b' c' : ℝ) (n : ℤ), 
      a' * (3/4)^2 + b' * (3/4) + c' = -25/16 ∧
      -b' / (2 * a') = 3/4 ∧
      a' + b' + c' = n := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2494_249461


namespace NUMINAMATH_CALUDE_binomial_12_choose_3_l2494_249434

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_3_l2494_249434


namespace NUMINAMATH_CALUDE_point_in_region_b_range_l2494_249450

theorem point_in_region_b_range (b : ℝ) :
  let P : ℝ × ℝ := (-1, 2)
  (2 * P.1 + 3 * P.2 - b > 0) → (b < 4) :=
by sorry

end NUMINAMATH_CALUDE_point_in_region_b_range_l2494_249450


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2494_249417

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y ≥ 1/a + 1/b) →
  1/a + 1/b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2494_249417


namespace NUMINAMATH_CALUDE_simplify_product_of_radicals_l2494_249455

theorem simplify_product_of_radicals (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (54 * x) * Real.sqrt (20 * x) * Real.sqrt (14 * x) = 12 * Real.sqrt (105 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_radicals_l2494_249455


namespace NUMINAMATH_CALUDE_largest_number_in_L_shape_l2494_249494

/-- Represents the different orientations of the "L" shape -/
inductive LShape
  | First  : LShape  -- (x-8, x-7, x)
  | Second : LShape  -- (x-7, x-6, x)
  | Third  : LShape  -- (x-7, x-1, x)
  | Fourth : LShape  -- (x-8, x-1, x)

/-- Calculates the sum of the three numbers in the "L" shape -/
def sumLShape (shape : LShape) (x : ℕ) : ℕ :=
  match shape with
  | LShape.First  => x - 8 + x - 7 + x
  | LShape.Second => x - 7 + x - 6 + x
  | LShape.Third  => x - 7 + x - 1 + x
  | LShape.Fourth => x - 8 + x - 1 + x

/-- The main theorem to be proved -/
theorem largest_number_in_L_shape : 
  ∃ (shape : LShape) (x : ℕ), sumLShape shape x = 2015 ∧ 
  (∀ (shape' : LShape) (y : ℕ), sumLShape shape' y = 2015 → y ≤ x) ∧ 
  x = 676 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_in_L_shape_l2494_249494


namespace NUMINAMATH_CALUDE_boy_escapes_l2494_249477

/-- Represents the square pool -/
structure Pool :=
  (side_length : ℝ)
  (boy_position : ℝ × ℝ)
  (teacher_position : ℝ × ℝ)

/-- Represents the speeds of the boy and teacher -/
structure Speeds :=
  (boy_swim : ℝ)
  (boy_run : ℝ)
  (teacher_run : ℝ)

/-- Checks if the boy can escape given the pool configuration and speeds -/
def can_escape (p : Pool) (s : Speeds) : Prop :=
  p.side_length = 2 ∧
  p.boy_position = (0, 0) ∧
  p.teacher_position = (1, 1) ∧
  s.boy_swim = s.teacher_run / 3 ∧
  s.boy_run > s.teacher_run

theorem boy_escapes (p : Pool) (s : Speeds) :
  can_escape p s → true :=
sorry

end NUMINAMATH_CALUDE_boy_escapes_l2494_249477


namespace NUMINAMATH_CALUDE_point_b_coordinate_l2494_249431

theorem point_b_coordinate (b : ℝ) : 
  (|(-2) - b| = 3) ↔ (b = -5 ∨ b = 1) := by sorry

end NUMINAMATH_CALUDE_point_b_coordinate_l2494_249431


namespace NUMINAMATH_CALUDE_zoo_field_trip_buses_l2494_249474

theorem zoo_field_trip_buses (fifth_graders sixth_graders seventh_graders : ℕ)
  (teachers_per_grade parents_per_grade : ℕ) (seats_per_bus : ℕ)
  (h1 : fifth_graders = 109)
  (h2 : sixth_graders = 115)
  (h3 : seventh_graders = 118)
  (h4 : teachers_per_grade = 4)
  (h5 : parents_per_grade = 2)
  (h6 : seats_per_bus = 72) :
  (fifth_graders + sixth_graders + seventh_graders +
   3 * (teachers_per_grade + parents_per_grade) + seats_per_bus - 1) / seats_per_bus = 5 :=
by sorry

end NUMINAMATH_CALUDE_zoo_field_trip_buses_l2494_249474


namespace NUMINAMATH_CALUDE_dollar_four_neg_one_l2494_249491

-- Define the $ operation
def dollar (x y : ℤ) : ℤ := x * (y + 2) + 2 * x * y

-- Theorem statement
theorem dollar_four_neg_one : dollar 4 (-1) = -4 := by
  sorry

end NUMINAMATH_CALUDE_dollar_four_neg_one_l2494_249491


namespace NUMINAMATH_CALUDE_range_of_g_l2494_249459

theorem range_of_g (x : ℝ) : 3/4 ≤ Real.cos x ^ 4 + Real.sin x ^ 2 ∧ Real.cos x ^ 4 + Real.sin x ^ 2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l2494_249459


namespace NUMINAMATH_CALUDE_hexagon_central_symmetry_l2494_249469

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon -/
structure Hexagon where
  vertices : Fin 6 → Point

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → Point

/-- Checks if a hexagon is centrally symmetric -/
def isCentrallySymmetric (h : Hexagon) : Prop := sorry

/-- Checks if a hexagon is regular -/
def isRegular (h : Hexagon) : Prop := sorry

/-- Constructs equilateral triangles on each side of the hexagon -/
def constructOutwardTriangles (h : Hexagon) : Fin 6 → EquilateralTriangle := sorry

/-- Finds the midpoints of the sides of the new hexagon formed by the triangle vertices -/
def findMidpoints (h : Hexagon) (triangles : Fin 6 → EquilateralTriangle) : Hexagon := sorry

/-- The main theorem -/
theorem hexagon_central_symmetry 
  (h : Hexagon) 
  (triangles : Fin 6 → EquilateralTriangle)
  (midpoints : Hexagon) 
  (h_triangles : triangles = constructOutwardTriangles h)
  (h_midpoints : midpoints = findMidpoints h triangles)
  (h_regular : isRegular midpoints) :
  isCentrallySymmetric h := sorry

end NUMINAMATH_CALUDE_hexagon_central_symmetry_l2494_249469


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2494_249480

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {5, 6, 7}

theorem complement_intersection_theorem : 
  (U \ A) ∩ (U \ B) = {4, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2494_249480


namespace NUMINAMATH_CALUDE_area_regular_dodecagon_formula_l2494_249449

/-- A regular dodecagon inscribed in a circle -/
structure RegularDodecagon (r : ℝ) where
  -- The radius of the circumscribed circle
  radius : ℝ
  radius_pos : radius > 0
  -- The dodecagon is regular and inscribed in the circle

/-- The area of a regular dodecagon -/
def area_regular_dodecagon (d : RegularDodecagon r) : ℝ := 3 * r^2

/-- Theorem: The area of a regular dodecagon inscribed in a circle with radius r is 3r² -/
theorem area_regular_dodecagon_formula (r : ℝ) (hr : r > 0) :
  ∀ (d : RegularDodecagon r), area_regular_dodecagon d = 3 * r^2 :=
by
  sorry

end NUMINAMATH_CALUDE_area_regular_dodecagon_formula_l2494_249449


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l2494_249488

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- Generates the nth pair in the sequence -/
def nthPair (n : ℕ) : IntPair :=
  sorry

/-- The sum of the numbers in a pair -/
def pairSum (p : IntPair) : ℕ :=
  p.first + p.second

/-- The number of pairs before the nth group -/
def pairsBeforeGroup (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the 60th pair is (5, 7) -/
theorem sixtieth_pair_is_five_seven :
  nthPair 60 = IntPair.mk 5 7 :=
sorry

end NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l2494_249488


namespace NUMINAMATH_CALUDE_function_inequality_range_l2494_249451

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem function_inequality_range 
  (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_mono : monotone_increasing_on f (Set.Ici 0))
  (h_value : f (-2) = 1) :
  {x : ℝ | f (x - 2) ≤ 1} = Set.Icc 0 4 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_range_l2494_249451


namespace NUMINAMATH_CALUDE_probability_positive_sum_is_one_third_l2494_249419

/-- The set of card values in the bag -/
def card_values : Finset Int := {-2, -1, 2}

/-- The sample space of all possible outcomes when drawing two cards with replacement -/
def sample_space : Finset (Int × Int) :=
  card_values.product card_values

/-- The set of favorable outcomes (sum of drawn cards is positive) -/
def favorable_outcomes : Finset (Int × Int) :=
  sample_space.filter (fun p => p.1 + p.2 > 0)

/-- The probability of drawing two cards with a positive sum -/
def probability_positive_sum : ℚ :=
  (favorable_outcomes.card : ℚ) / (sample_space.card : ℚ)

theorem probability_positive_sum_is_one_third :
  probability_positive_sum = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_positive_sum_is_one_third_l2494_249419


namespace NUMINAMATH_CALUDE_friends_picnic_only_l2494_249415

/-- Given information about friends meeting for different activities, 
    prove that the number of friends meeting for picnic only is 20. -/
theorem friends_picnic_only (total : ℕ) (movie : ℕ) (games : ℕ) 
  (movie_picnic : ℕ) (movie_games : ℕ) (picnic_games : ℕ) (all_three : ℕ) :
  total = 31 ∧ 
  movie = 10 ∧ 
  games = 5 ∧ 
  movie_picnic = 4 ∧ 
  movie_games = 2 ∧ 
  picnic_games = 0 ∧ 
  all_three = 2 → 
  ∃ (movie_only picnic_only games_only : ℕ),
    total = movie_only + picnic_only + games_only + movie_picnic + movie_games + picnic_games + all_three ∧
    movie = movie_only + movie_picnic + movie_games + all_three ∧
    games = games_only + movie_games + all_three ∧
    picnic_only = 20 := by
  sorry

end NUMINAMATH_CALUDE_friends_picnic_only_l2494_249415


namespace NUMINAMATH_CALUDE_fewer_green_marbles_percentage_l2494_249442

/-- Proves that the percentage of fewer green marbles compared to yellow marbles is 50% -/
theorem fewer_green_marbles_percentage (total : ℕ) (white yellow green red : ℕ) :
  total = 50 ∧
  white = total / 2 ∧
  yellow = 12 ∧
  red = 7 ∧
  green = total - (white + yellow + red) →
  (yellow - green) / yellow * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_fewer_green_marbles_percentage_l2494_249442


namespace NUMINAMATH_CALUDE_tourist_group_size_l2494_249411

theorem tourist_group_size (k : ℕ) : 
  (∃ n : ℕ, n > 0 ∧ 2 * k ≡ 1 [MOD n] ∧ 3 * k ≡ 13 [MOD n]) → 
  (∃ n : ℕ, n = 23 ∧ 2 * k ≡ 1 [MOD n] ∧ 3 * k ≡ 13 [MOD n]) :=
by sorry

end NUMINAMATH_CALUDE_tourist_group_size_l2494_249411


namespace NUMINAMATH_CALUDE_salmon_trip_count_l2494_249496

/-- The number of male salmon that returned to their rivers -/
def male_salmon : ℕ := 712261

/-- The number of female salmon that returned to their rivers -/
def female_salmon : ℕ := 259378

/-- The total number of salmon that made the trip -/
def total_salmon : ℕ := male_salmon + female_salmon

theorem salmon_trip_count : total_salmon = 971639 := by
  sorry

end NUMINAMATH_CALUDE_salmon_trip_count_l2494_249496


namespace NUMINAMATH_CALUDE_point_on_line_l2494_249482

/-- A complex number z represented as (a-1) + 3i, where a is a real number -/
def z (a : ℝ) : ℂ := Complex.mk (a - 1) 3

/-- The line y = x + 2 in the complex plane -/
def line (x : ℝ) : ℝ := x + 2

/-- Theorem: If z(a) is on the line y = x + 2, then a = 2 -/
theorem point_on_line (a : ℝ) : z a = Complex.mk (z a).re (line (z a).re) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2494_249482


namespace NUMINAMATH_CALUDE_linear_control_periodic_bound_l2494_249453

/-- A function f: ℝ → ℝ is a linear control function if |f'(x)| ≤ 1 for all x ∈ ℝ -/
def LinearControlFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, DifferentiableAt ℝ f x ∧ |deriv f x| ≤ 1

/-- A function f: ℝ → ℝ is periodic with period T if f(x + T) = f(x) for all x ∈ ℝ -/
def Periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

theorem linear_control_periodic_bound 
    (f : ℝ → ℝ) (T : ℝ) 
    (h1 : LinearControlFunction f)
    (h2 : StrictMono f)
    (h3 : Periodic f T) :
    ∀ x₁ x₂ : ℝ, |f x₁ - f x₂| ≤ T := by
  sorry


end NUMINAMATH_CALUDE_linear_control_periodic_bound_l2494_249453


namespace NUMINAMATH_CALUDE_flagpole_break_height_approx_l2494_249490

/-- The height of the flagpole in meters -/
def flagpole_height : ℝ := 5

/-- The distance from the base of the flagpole to where the broken part touches the ground, in meters -/
def ground_distance : ℝ := 1

/-- The approximate height where the flagpole breaks, in meters -/
def break_height : ℝ := 2.4

/-- Theorem stating that the break height is approximately correct -/
theorem flagpole_break_height_approx :
  let total_height := flagpole_height
  let distance := ground_distance
  let break_point := break_height
  abs (break_point - (total_height * distance / (2 * total_height))) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_flagpole_break_height_approx_l2494_249490


namespace NUMINAMATH_CALUDE_mount_everest_temperature_difference_l2494_249416

/-- Temperature difference between two points -/
def temperature_difference (t1 : ℝ) (t2 : ℝ) : ℝ := t1 - t2

/-- Temperature at the foot of Mount Everest in °C -/
def foot_temperature : ℝ := 24

/-- Temperature at the summit of Mount Everest in °C -/
def summit_temperature : ℝ := -50

/-- Theorem stating the temperature difference between the foot and summit of Mount Everest -/
theorem mount_everest_temperature_difference :
  temperature_difference foot_temperature summit_temperature = 74 := by
  sorry

end NUMINAMATH_CALUDE_mount_everest_temperature_difference_l2494_249416


namespace NUMINAMATH_CALUDE_prime_power_difference_l2494_249440

theorem prime_power_difference (n : ℕ) (p : ℕ) (k : ℕ) 
  (h1 : n > 0) 
  (h2 : Nat.Prime p) 
  (h3 : 3^n - 2^n = p^k) : 
  Nat.Prime n := by
sorry

end NUMINAMATH_CALUDE_prime_power_difference_l2494_249440


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2494_249402

theorem perfect_square_trinomial (a b : ℝ) : a^2 + 6*a*b + 9*b^2 = (a + 3*b)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2494_249402


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2494_249448

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 5) / 8 = 5 / (x - 8) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2494_249448


namespace NUMINAMATH_CALUDE_equation_equivalence_l2494_249418

theorem equation_equivalence (x y : ℝ) :
  x^2 * (y + y^2) = y^3 + x^4 ↔ y = x ∨ y = -x ∨ y = x^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2494_249418


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2494_249414

/-- A quadratic function f(x) = x^2 + bx + c with f(1) = 0 and f(3) = 0 satisfies f(-1) = 8 -/
theorem quadratic_function_property (b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 + b*x + c) 
  (h2 : f 1 = 0) 
  (h3 : f 3 = 0) : 
  f (-1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2494_249414


namespace NUMINAMATH_CALUDE_cone_volume_increase_l2494_249445

theorem cone_volume_increase (R H : ℝ) (hR : R = 5) (hH : H = 12) :
  ∃ y : ℝ, y > 0 ∧ (1 / 3) * π * (R + y)^2 * H = (1 / 3) * π * R^2 * (H + y) ∧ y = 31 / 12 :=
sorry

end NUMINAMATH_CALUDE_cone_volume_increase_l2494_249445


namespace NUMINAMATH_CALUDE_power_of_81_three_fourths_l2494_249466

theorem power_of_81_three_fourths : (81 : ℝ) ^ (3/4 : ℝ) = 27 := by sorry

end NUMINAMATH_CALUDE_power_of_81_three_fourths_l2494_249466


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2494_249495

theorem polynomial_factorization (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^2 - x - 1) * q x = (-987 * x^18 + 2584 * x^17 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2494_249495


namespace NUMINAMATH_CALUDE_pie_cost_satisfies_conditions_l2494_249413

/-- The cost of one pie in rubles -/
def pie_cost : ℚ := 20

/-- The total value of Masha's two-ruble coins -/
def two_ruble_coins : ℚ := 4 * pie_cost - 60

/-- The total value of Masha's five-ruble coins -/
def five_ruble_coins : ℚ := 5 * pie_cost - 60

/-- Theorem stating that the pie cost satisfies all given conditions -/
theorem pie_cost_satisfies_conditions :
  (4 * pie_cost = two_ruble_coins + 60) ∧
  (5 * pie_cost = five_ruble_coins + 60) ∧
  (6 * pie_cost = two_ruble_coins + five_ruble_coins + 60) :=
by sorry

#check pie_cost_satisfies_conditions

end NUMINAMATH_CALUDE_pie_cost_satisfies_conditions_l2494_249413


namespace NUMINAMATH_CALUDE_payment_is_two_l2494_249428

def payment_per_window (stories : ℕ) (windows_per_floor : ℕ) (subtraction_rate : ℚ)
  (days_taken : ℕ) (final_payment : ℚ) : ℚ :=
  let total_windows := stories * windows_per_floor
  let subtraction := (days_taken / 3 : ℚ) * subtraction_rate
  let original_payment := final_payment + subtraction
  original_payment / total_windows

theorem payment_is_two :
  payment_per_window 3 3 1 6 16 = 2 := by sorry

end NUMINAMATH_CALUDE_payment_is_two_l2494_249428


namespace NUMINAMATH_CALUDE_function_inequality_l2494_249463

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h1 : ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y)
variable (h2 : ∀ x, f (x + 2) = f (2 - x))

-- State the theorem
theorem function_inequality : f 2.5 > f 1 ∧ f 1 > f 3.5 := by sorry

end NUMINAMATH_CALUDE_function_inequality_l2494_249463


namespace NUMINAMATH_CALUDE_staircase_perimeter_l2494_249473

/-- A staircase-shaped region with specific properties -/
structure StaircaseRegion where
  num_sides : ℕ
  side_length : ℝ
  total_area : ℝ

/-- The perimeter of a StaircaseRegion -/
def perimeter (r : StaircaseRegion) : ℝ := sorry

/-- Theorem stating the perimeter of a specific StaircaseRegion -/
theorem staircase_perimeter :
  ∀ (r : StaircaseRegion),
    r.num_sides = 12 ∧
    r.side_length = 1 ∧
    r.total_area = 120 →
    perimeter r = 36 := by sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l2494_249473


namespace NUMINAMATH_CALUDE_y_equals_five_l2494_249452

/-- Configuration of numbers in a triangular arrangement -/
structure NumberTriangle where
  y : ℝ
  z : ℝ
  second_row : ℝ
  third_row : ℝ
  h1 : second_row = y * 10
  h2 : third_row = second_row * z

/-- The value of y in the given configuration is 5 -/
theorem y_equals_five (t : NumberTriangle) (h3 : t.second_row = 50) (h4 : t.third_row = 300) : t.y = 5 := by
  sorry


end NUMINAMATH_CALUDE_y_equals_five_l2494_249452


namespace NUMINAMATH_CALUDE_cave_depth_remaining_l2494_249410

theorem cave_depth_remaining (total_depth : ℕ) (traveled_distance : ℕ) 
  (h1 : total_depth = 1218)
  (h2 : traveled_distance = 849) :
  total_depth - traveled_distance = 369 := by
sorry

end NUMINAMATH_CALUDE_cave_depth_remaining_l2494_249410


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l2494_249470

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 130) 
  (h2 : x*y = 45) : 
  x + y ≤ 2 * Real.sqrt 55 := by
sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l2494_249470


namespace NUMINAMATH_CALUDE_M_intersect_N_l2494_249478

def M : Set ℕ := {1, 2, 4, 8}

def N : Set ℕ := {x : ℕ | ∃ k : ℕ, x = 2 * k}

theorem M_intersect_N : M ∩ N = {2, 4, 8} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l2494_249478


namespace NUMINAMATH_CALUDE_age_ratio_l2494_249426

theorem age_ratio (sachin_age rahul_age : ℕ) : 
  sachin_age = 49 → 
  rahul_age = sachin_age + 14 → 
  (sachin_age : ℚ) / rahul_age = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_l2494_249426


namespace NUMINAMATH_CALUDE_inequality_proof_l2494_249403

theorem inequality_proof (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (2 + a) * (2 + b) ≥ c * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2494_249403


namespace NUMINAMATH_CALUDE_tristan_study_schedule_l2494_249443

/-- Tristan's study schedule problem -/
theorem tristan_study_schedule (monday tuesday wednesday thursday friday goal saturday sunday : ℝ) 
  (h1 : monday = 4)
  (h2 : tuesday = 5)
  (h3 : wednesday = 6)
  (h4 : thursday = tuesday / 2)
  (h5 : friday = 2 * monday)
  (h6 : goal = 41.5)
  (h7 : saturday = sunday)
  (h8 : monday + tuesday + wednesday + thursday + friday + saturday + sunday = goal) :
  saturday = 8 := by
sorry


end NUMINAMATH_CALUDE_tristan_study_schedule_l2494_249443


namespace NUMINAMATH_CALUDE_jane_baskets_l2494_249498

/-- The number of baskets Jane used to sort her apples -/
def num_baskets : ℕ := sorry

/-- The total number of apples Jane picked -/
def total_apples : ℕ := 64

/-- The number of apples in each basket after Jane's sister took some -/
def apples_per_basket_after : ℕ := 13

/-- The number of apples Jane's sister took from each basket -/
def apples_taken_per_basket : ℕ := 3

theorem jane_baskets : 
  (num_baskets * (apples_per_basket_after + apples_taken_per_basket) = total_apples) ∧
  (num_baskets = 4) := by sorry

end NUMINAMATH_CALUDE_jane_baskets_l2494_249498


namespace NUMINAMATH_CALUDE_system_solution_l2494_249408

/-- Given a system of equations in x, y, and m, prove the relationship between x and y,
    and find the value of m when x + y = -10 -/
theorem system_solution (x y m : ℝ) 
  (eq1 : 3 * x + 5 * y = m + 2)
  (eq2 : 2 * x + 3 * y = m) :
  (y = 1 - x / 2) ∧ 
  (x + y = -10 → m = -8) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2494_249408


namespace NUMINAMATH_CALUDE_range_of_m_plus_n_l2494_249436

/-- The function f(x) -/
noncomputable def f (m n x : ℝ) : ℝ := m * Real.exp x + x^2 + n * x

/-- The set of roots of f(x) -/
def roots (m n : ℝ) : Set ℝ := {x | f m n x = 0}

/-- The set of roots of f(f(x)) -/
def double_roots (m n : ℝ) : Set ℝ := {x | f m n (f m n x) = 0}

/-- Main theorem: Given f(x) = me^x + x^2 + nx, where the roots of f and f(f) are the same and non-empty,
    the range of m+n is [0, 4) -/
theorem range_of_m_plus_n (m n : ℝ) 
    (h1 : roots m n = double_roots m n) 
    (h2 : roots m n ≠ ∅) : 
    0 ≤ m + n ∧ m + n < 4 := by sorry

end NUMINAMATH_CALUDE_range_of_m_plus_n_l2494_249436


namespace NUMINAMATH_CALUDE_table_tennis_arrangements_l2494_249475

def total_players : ℕ := 10
def main_players : ℕ := 3
def match_players : ℕ := 5
def remaining_players : ℕ := total_players - main_players

theorem table_tennis_arrangements :
  (main_players.factorial) * (remaining_players.choose 2) = 252 :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_arrangements_l2494_249475


namespace NUMINAMATH_CALUDE_probability_triangle_or_hexagon_l2494_249497

theorem probability_triangle_or_hexagon :
  let total_figures : ℕ := 10
  let triangles : ℕ := 3
  let squares : ℕ := 4
  let circles : ℕ := 2
  let hexagons : ℕ := 1
  let target_figures := triangles + hexagons
  (target_figures : ℚ) / total_figures = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_triangle_or_hexagon_l2494_249497


namespace NUMINAMATH_CALUDE_min_toothpicks_removal_l2494_249407

/-- Represents a geometric figure made of toothpicks -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  triangles : ℕ
  squares : ℕ

/-- Calculates the minimum number of toothpicks to remove to eliminate all geometric figures -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ :=
  sorry

/-- Theorem stating the minimum number of toothpicks to remove for the given figure -/
theorem min_toothpicks_removal (figure : ToothpickFigure) 
  (h1 : figure.total_toothpicks = 50)
  (h2 : figure.triangles = 10)
  (h3 : figure.squares = 4) :
  min_toothpicks_to_remove figure = 10 :=
sorry

end NUMINAMATH_CALUDE_min_toothpicks_removal_l2494_249407


namespace NUMINAMATH_CALUDE_hugo_rolls_six_given_win_l2494_249487

-- Define the number of players and sides on the die
def num_players : ℕ := 5
def num_sides : ℕ := 8

-- Define the event of Hugo winning
def hugo_wins : Set (Fin num_players → Fin num_sides) := sorry

-- Define the event of Hugo rolling a 6 on his first roll
def hugo_rolls_six : Set (Fin num_players → Fin num_sides) := sorry

-- Define the probability measure
noncomputable def P : Set (Fin num_players → Fin num_sides) → ℝ := sorry

-- Theorem statement
theorem hugo_rolls_six_given_win :
  P (hugo_rolls_six ∩ hugo_wins) / P hugo_wins = 6375 / 32768 := by sorry

end NUMINAMATH_CALUDE_hugo_rolls_six_given_win_l2494_249487


namespace NUMINAMATH_CALUDE_rectangle_y_coordinate_sum_l2494_249430

/-- Given a rectangle with opposite vertices (5,22) and (12,-3),
    the sum of the y-coordinates of the other two vertices is 19. -/
theorem rectangle_y_coordinate_sum :
  let v1 : ℝ × ℝ := (5, 22)
  let v2 : ℝ × ℝ := (12, -3)
  let mid_y : ℝ := (v1.2 + v2.2) / 2
  19 = 2 * mid_y := by sorry

end NUMINAMATH_CALUDE_rectangle_y_coordinate_sum_l2494_249430


namespace NUMINAMATH_CALUDE_total_marbles_l2494_249404

theorem total_marbles (jungkook_marbles : ℕ) (jimin_extra_marbles : ℕ) : 
  jungkook_marbles = 3 → 
  jimin_extra_marbles = 4 → 
  jungkook_marbles + (jungkook_marbles + jimin_extra_marbles) = 10 := by
sorry

end NUMINAMATH_CALUDE_total_marbles_l2494_249404


namespace NUMINAMATH_CALUDE_min_draws_for_even_product_l2494_249456

theorem min_draws_for_even_product (S : Finset ℕ) : 
  S = Finset.range 16 →
  (∃ n : ℕ, n ∈ S ∧ Even n) →
  (∀ T ⊆ S, T.card = 9 → ∃ m ∈ T, Even m) ∧
  (∃ U ⊆ S, U.card = 8 ∧ ∀ k ∈ U, ¬Even k) :=
by sorry

end NUMINAMATH_CALUDE_min_draws_for_even_product_l2494_249456


namespace NUMINAMATH_CALUDE_middle_card_is_six_l2494_249427

/-- Represents a set of three cards with positive integers -/
structure CardSet where
  left : Nat
  middle : Nat
  right : Nat
  sum_is_17 : left + middle + right = 17
  increasing : left < middle ∧ middle < right

/-- Predicate to check if a number allows for multiple possibilities when seen on the left -/
def leftIndeterminate (n : Nat) : Prop :=
  ∃ (cs1 cs2 : CardSet), cs1.left = n ∧ cs2.left = n ∧ cs1 ≠ cs2

/-- Predicate to check if a number allows for multiple possibilities when seen on the right -/
def rightIndeterminate (n : Nat) : Prop :=
  ∃ (cs1 cs2 : CardSet), cs1.right = n ∧ cs2.right = n ∧ cs1 ≠ cs2

/-- Predicate to check if a number allows for multiple possibilities when seen in the middle -/
def middleIndeterminate (n : Nat) : Prop :=
  ∃ (cs1 cs2 : CardSet), cs1.middle = n ∧ cs2.middle = n ∧ cs1 ≠ cs2

/-- The main theorem stating that the middle card must be 6 -/
theorem middle_card_is_six :
  ∀ (cs : CardSet),
    leftIndeterminate cs.left →
    rightIndeterminate cs.right →
    middleIndeterminate cs.middle →
    cs.middle = 6 := by
  sorry

end NUMINAMATH_CALUDE_middle_card_is_six_l2494_249427


namespace NUMINAMATH_CALUDE_satisfying_polynomial_iff_polynomial_form_l2494_249439

/-- A polynomial that satisfies the given equation for all real x -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (x^2 - 6*x + 8) * P x = (x^2 + 2*x) * P (x - 2)

/-- The form of the polynomial that satisfies the equation -/
def PolynomialForm (P : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x : ℝ, P x = c * x^2 * (x^2 - 4)

/-- Theorem stating the equivalence between satisfying the equation and having the specific form -/
theorem satisfying_polynomial_iff_polynomial_form :
  ∀ P : ℝ → ℝ, SatisfyingPolynomial P ↔ PolynomialForm P :=
sorry

end NUMINAMATH_CALUDE_satisfying_polynomial_iff_polynomial_form_l2494_249439
