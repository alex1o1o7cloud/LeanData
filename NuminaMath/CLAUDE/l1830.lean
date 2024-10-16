import Mathlib

namespace NUMINAMATH_CALUDE_weeks_to_save_for_bike_l1830_183081

def bike_cost : ℕ := 640
def birthday_money : ℕ := 60 + 40 + 20
def weekly_earnings : ℕ := 20

theorem weeks_to_save_for_bike :
  ∃ (weeks : ℕ), birthday_money + weeks * weekly_earnings = bike_cost ∧ weeks = 26 := by
  sorry

end NUMINAMATH_CALUDE_weeks_to_save_for_bike_l1830_183081


namespace NUMINAMATH_CALUDE_pirate_treasure_l1830_183029

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by sorry

end NUMINAMATH_CALUDE_pirate_treasure_l1830_183029


namespace NUMINAMATH_CALUDE_inequality_range_l1830_183001

theorem inequality_range (a : ℝ) : 
  (∀ x y, x ∈ Set.Icc 0 (π/6) → y ∈ Set.Ioi 0 → 
    y/4 - 2*(Real.cos x)^2 ≥ a*(Real.sin x) - 9/y) → 
  a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l1830_183001


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1830_183087

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y + m = 0 → y = x) → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1830_183087


namespace NUMINAMATH_CALUDE_max_distance_point_circle_l1830_183073

/-- The maximum distance between a point and a circle -/
theorem max_distance_point_circle :
  let center : ℝ × ℝ := (1, 2)
  let radius : ℝ := 2
  let P : ℝ × ℝ := (3, 3)
  (∀ M : ℝ × ℝ, (M.1 - center.1)^2 + (M.2 - center.2)^2 = radius^2 →
    Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) ≤ Real.sqrt 5 + 2) ∧
  (∃ M : ℝ × ℝ, (M.1 - center.1)^2 + (M.2 - center.2)^2 = radius^2 ∧
    Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) = Real.sqrt 5 + 2) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_point_circle_l1830_183073


namespace NUMINAMATH_CALUDE_t_leq_s_l1830_183008

theorem t_leq_s (a b : ℝ) (t s : ℝ) (ht : t = a + 2*b) (hs : s = a + b^2 + 1) : t ≤ s := by
  sorry

end NUMINAMATH_CALUDE_t_leq_s_l1830_183008


namespace NUMINAMATH_CALUDE_abs_minus_sqrt_eq_three_l1830_183048

theorem abs_minus_sqrt_eq_three (a : ℝ) (h : a < 0) : |a - 3| - Real.sqrt (a^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_minus_sqrt_eq_three_l1830_183048


namespace NUMINAMATH_CALUDE_q_factor_change_l1830_183006

theorem q_factor_change (w m z : ℝ) (hw : w ≠ 0) (hm : m ≠ 0) (hz : z ≠ 0) :
  let q := 5 * w / (4 * m * z^2)
  let q_new := 5 * (4*w) / (4 * (2*m) * (3*z)^2)
  q_new = (2/9) * q := by
sorry

end NUMINAMATH_CALUDE_q_factor_change_l1830_183006


namespace NUMINAMATH_CALUDE_probability_sum_multiple_of_three_l1830_183092

def die_faces : ℕ := 6

def total_outcomes : ℕ := die_faces * die_faces

def is_multiple_of_three (n : ℕ) : Prop := ∃ k, n = 3 * k

def favorable_outcomes : ℕ := 12

theorem probability_sum_multiple_of_three :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_multiple_of_three_l1830_183092


namespace NUMINAMATH_CALUDE_parabola_perpendicular_range_l1830_183002

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Parabola equation y^2 = x + 4 -/
def OnParabola (p : Point) : Prop :=
  p.y^2 = p.x + 4

/-- Perpendicular lines have product of slopes equal to -1 -/
def Perpendicular (a b c : Point) : Prop :=
  (b.y - a.y) * (c.y - b.y) = -(b.x - a.x) * (c.x - b.x)

/-- The main theorem -/
theorem parabola_perpendicular_range :
  ∀ (b c : Point),
    OnParabola b → OnParabola c →
    Perpendicular ⟨0, 2⟩ b c →
    c.y ≤ 0 ∨ c.y ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_perpendicular_range_l1830_183002


namespace NUMINAMATH_CALUDE_second_difference_quadratic_constant_second_difference_implies_A_second_difference_one_implies_A_half_second_difference_seven_implies_A_seven_half_l1830_183020

/-- Second difference of a function f at point n -/
def secondDifference (f : ℕ → ℚ) (n : ℕ) : ℚ :=
  f (n + 2) - 2 * f (n + 1) + f n

/-- Quadratic function with rational coefficients -/
def quadraticFunction (A B C : ℚ) (n : ℕ) : ℚ :=
  A * n^2 + B * n + C

theorem second_difference_quadratic (A B C : ℚ) :
  ∀ n : ℕ, secondDifference (quadraticFunction A B C) n = 2 * A :=
sorry

theorem constant_second_difference_implies_A (A B C k : ℚ) :
  (∀ n : ℕ, secondDifference (quadraticFunction A B C) n = k) → A = k / 2 :=
sorry

theorem second_difference_one_implies_A_half :
  ∀ A B C : ℚ,
  (∀ n : ℕ, secondDifference (quadraticFunction A B C) n = 1) →
  A = 1 / 2 :=
sorry

theorem second_difference_seven_implies_A_seven_half :
  ∀ A B C : ℚ,
  (∀ n : ℕ, secondDifference (quadraticFunction A B C) n = 7) →
  A = 7 / 2 :=
sorry

end NUMINAMATH_CALUDE_second_difference_quadratic_constant_second_difference_implies_A_second_difference_one_implies_A_half_second_difference_seven_implies_A_seven_half_l1830_183020


namespace NUMINAMATH_CALUDE_linear_decreasing_iff_negative_slope_l1830_183025

/-- A linear function from ℝ to ℝ -/
def LinearFunction (m b : ℝ) : ℝ → ℝ := fun x ↦ m * x + b

/-- A function is decreasing if for any x₁ < x₂, f(x₁) > f(x₂) -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

theorem linear_decreasing_iff_negative_slope (m b : ℝ) :
  IsDecreasing (LinearFunction m b) ↔ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_decreasing_iff_negative_slope_l1830_183025


namespace NUMINAMATH_CALUDE_inductive_reasoning_correct_l1830_183072

-- Define the types of reasoning
inductive ReasoningMethod
| Analogical
| Deductive
| Inductive
| Reasonable

-- Define the direction of reasoning
inductive ReasoningDirection
| IndividualToIndividual
| GeneralToSpecific
| IndividualToGeneral
| Other

-- Define a function that describes the direction of each reasoning method
def reasoningDirection (method : ReasoningMethod) : ReasoningDirection :=
  match method with
  | ReasoningMethod.Analogical => ReasoningDirection.IndividualToIndividual
  | ReasoningMethod.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningMethod.Inductive => ReasoningDirection.IndividualToGeneral
  | ReasoningMethod.Reasonable => ReasoningDirection.Other

-- Define a predicate for whether a reasoning method can be used in a proof
def canBeUsedInProof (method : ReasoningMethod) : Prop :=
  match method with
  | ReasoningMethod.Reasonable => False
  | _ => True

-- Theorem stating that inductive reasoning is the only correct answer
theorem inductive_reasoning_correct :
  (∀ m : ReasoningMethod, m ≠ ReasoningMethod.Inductive →
    (reasoningDirection m ≠ ReasoningDirection.IndividualToGeneral ∨
     ¬canBeUsedInProof m)) ∧
  (reasoningDirection ReasoningMethod.Inductive = ReasoningDirection.IndividualToGeneral ∧
   canBeUsedInProof ReasoningMethod.Inductive) :=
by
  sorry


end NUMINAMATH_CALUDE_inductive_reasoning_correct_l1830_183072


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1830_183070

theorem quadratic_inequality_solution (x : ℝ) :
  (2 * x - 1) * (3 * x + 1) > 0 ↔ x < -1/3 ∨ x > 1/2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1830_183070


namespace NUMINAMATH_CALUDE_wall_thickness_calculation_l1830_183064

/-- Calculates the thickness of a wall given brick dimensions and wall specifications -/
theorem wall_thickness_calculation (brick_length brick_width brick_height : ℝ)
                                   (wall_length wall_height : ℝ)
                                   (num_bricks : ℕ) :
  brick_length = 50 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_length = 800 →
  wall_height = 600 →
  num_bricks = 3200 →
  ∃ (wall_thickness : ℝ),
    wall_thickness = 22.5 ∧
    wall_length * wall_height * wall_thickness = num_bricks * brick_length * brick_width * brick_height :=
by
  sorry

#check wall_thickness_calculation

end NUMINAMATH_CALUDE_wall_thickness_calculation_l1830_183064


namespace NUMINAMATH_CALUDE_p_and_q_true_l1830_183055

theorem p_and_q_true (h : ¬(¬(p ∧ q))) : p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_true_l1830_183055


namespace NUMINAMATH_CALUDE_production_value_range_l1830_183047

-- Define the production value function
def f (x : ℝ) : ℝ := x * (220 - 2 * x)

-- Define the theorem
theorem production_value_range :
  ∀ x : ℝ, f x ≥ 6000 ↔ 50 < x ∧ x < 60 :=
by sorry

end NUMINAMATH_CALUDE_production_value_range_l1830_183047


namespace NUMINAMATH_CALUDE_two_thousand_twentieth_digit_l1830_183093

/-- Represents the decimal number formed by concatenating integers from 1 to 1000 -/
def x : ℚ := sorry

/-- Returns the nth digit after the decimal point in the number x -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- The 2020th digit after the decimal point in x is 7 -/
theorem two_thousand_twentieth_digit : nth_digit 2020 = 7 := by sorry

end NUMINAMATH_CALUDE_two_thousand_twentieth_digit_l1830_183093


namespace NUMINAMATH_CALUDE_carls_membership_number_l1830_183076

/-- A predicate to check if a number is a two-digit prime -/
def isTwoDigitPrime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

/-- The main theorem -/
theorem carls_membership_number
  (a b c d : ℕ)
  (ha : isTwoDigitPrime a)
  (hb : isTwoDigitPrime b)
  (hc : isTwoDigitPrime c)
  (hd : isTwoDigitPrime d)
  (sum_all : a + b + c + d = 100)
  (sum_no_ben : b + c + d = 30)
  (sum_no_carl : a + b + d = 29)
  (sum_no_david : a + b + c = 23) :
  c = 23 := by
  sorry


end NUMINAMATH_CALUDE_carls_membership_number_l1830_183076


namespace NUMINAMATH_CALUDE_sum_remainder_mod_8_l1830_183069

theorem sum_remainder_mod_8 : (7150 + 7151 + 7152 + 7153 + 7154 + 7155) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_8_l1830_183069


namespace NUMINAMATH_CALUDE_soup_bins_calculation_l1830_183096

/-- Given a canned food drive with different types of food, calculate the number of bins of soup. -/
theorem soup_bins_calculation (total_bins vegetables_bins pasta_bins : ℝ) 
  (h1 : vegetables_bins = 0.125)
  (h2 : pasta_bins = 0.5)
  (h3 : total_bins = 0.75) :
  total_bins - (vegetables_bins + pasta_bins) = 0.125 := by
  sorry

#check soup_bins_calculation

end NUMINAMATH_CALUDE_soup_bins_calculation_l1830_183096


namespace NUMINAMATH_CALUDE_primle_is_79_l1830_183091

def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_in_tens_place (n : ℕ) (d : ℕ) : Prop := n / 10 = d

def digit_in_ones_place (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem primle_is_79 (primle : ℕ) 
  (h1 : is_prime primle)
  (h2 : is_two_digit primle)
  (h3 : digit_in_tens_place primle 7)
  (h4 : ¬ digit_in_ones_place primle 7)
  (h5 : ¬ (digit_in_tens_place primle 1 ∨ digit_in_ones_place primle 1))
  (h6 : ¬ (digit_in_tens_place primle 3 ∨ digit_in_ones_place primle 3))
  (h7 : ¬ (digit_in_tens_place primle 4 ∨ digit_in_ones_place primle 4)) :
  primle = 79 := by
sorry

end NUMINAMATH_CALUDE_primle_is_79_l1830_183091


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1830_183032

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = f (f x) + f (y^2) + 2 * f (x * y)

/-- The theorem stating that the only functions satisfying the equation are f(x) = 0 or f(x) = x² -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (∀ x, f x = 0) ∨ (∀ x, f x = x^2) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1830_183032


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l1830_183051

/-- Given that when a person weighing 87 kg replaces a person weighing 67 kg,
    the average weight increases by 2.5 kg, prove that the number of persons
    initially is 8. -/
theorem initial_number_of_persons : ℕ :=
  let old_weight := 67
  let new_weight := 87
  let average_increase := 2.5
  let n := (new_weight - old_weight) / average_increase
  8

#check initial_number_of_persons

end NUMINAMATH_CALUDE_initial_number_of_persons_l1830_183051


namespace NUMINAMATH_CALUDE_fish_filets_count_l1830_183019

/-- The number of fish filets Ben and his family will have after their fishing trip -/
def fish_filets : ℕ :=
  let ben_fish := 4
  let judy_fish := 1
  let billy_fish := 3
  let jim_fish := 2
  let susie_fish := 5
  let small_fish := 3
  let filets_per_fish := 2
  let total_caught := ben_fish + judy_fish + billy_fish + jim_fish + susie_fish
  let kept_fish := total_caught - small_fish
  kept_fish * filets_per_fish

/-- Theorem stating that the number of fish filets Ben and his family will have is 24 -/
theorem fish_filets_count : fish_filets = 24 := by
  sorry

end NUMINAMATH_CALUDE_fish_filets_count_l1830_183019


namespace NUMINAMATH_CALUDE_f_equals_g_l1830_183004

-- Define the functions
def f (x : ℝ) : ℝ := x^2 - 2*x
def g (t : ℝ) : ℝ := t^2 - 2*t

-- State the theorem
theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l1830_183004


namespace NUMINAMATH_CALUDE_product_of_square_roots_l1830_183097

theorem product_of_square_roots (m : ℝ) :
  Real.sqrt (15 * m) * Real.sqrt (3 * m^2) * Real.sqrt (8 * m^3) = 6 * m^3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l1830_183097


namespace NUMINAMATH_CALUDE_line_quadrants_l1830_183026

/-- A line passing through the second and fourth quadrants has a negative slope -/
def passes_through_second_and_fourth_quadrants (k : ℝ) : Prop :=
  k < 0

/-- A line y = kx + b passes through the first and third quadrants if k > 0 -/
def passes_through_first_and_third_quadrants (k : ℝ) : Prop :=
  k > 0

/-- A line y = kx + b passes through the fourth quadrant if k > 0 and b < 0 -/
def passes_through_fourth_quadrant (k b : ℝ) : Prop :=
  k > 0 ∧ b < 0

theorem line_quadrants (k : ℝ) :
  passes_through_second_and_fourth_quadrants k →
  passes_through_first_and_third_quadrants (-k) ∧
  passes_through_fourth_quadrant (-k) (-1) :=
by sorry

end NUMINAMATH_CALUDE_line_quadrants_l1830_183026


namespace NUMINAMATH_CALUDE_power_product_squared_l1830_183011

theorem power_product_squared (m n : ℝ) : (m * n)^2 = m^2 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_squared_l1830_183011


namespace NUMINAMATH_CALUDE_soccer_season_games_l1830_183027

/-- The number of months in the soccer season -/
def season_length : ℕ := 3

/-- The number of soccer games played per month -/
def games_per_month : ℕ := 9

/-- The total number of soccer games played during the season -/
def total_games : ℕ := season_length * games_per_month

theorem soccer_season_games : total_games = 27 := by
  sorry

end NUMINAMATH_CALUDE_soccer_season_games_l1830_183027


namespace NUMINAMATH_CALUDE_min_square_difference_of_roots_l1830_183066

theorem min_square_difference_of_roots (α β b : ℝ) : 
  α^2 + 2*b*α + b = 1 → β^2 + 2*b*β + b = 1 → 
  ∀ γ δ c : ℝ, (γ^2 + 2*c*γ + c = 1 ∧ δ^2 + 2*c*δ + c = 1) → 
  (α - β)^2 ≥ 3 ∧ (∃ e : ℝ, (α - β)^2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_square_difference_of_roots_l1830_183066


namespace NUMINAMATH_CALUDE_perpendicular_planes_l1830_183018

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (a b : Line) 
  (α β : Plane) 
  (hab : a ≠ b) 
  (hαβ : α ≠ β) 
  (hab_perp : perp_line a b) 
  (haα_perp : perp_line_plane a α) 
  (hbβ_perp : perp_line_plane b β) : 
  perp_plane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l1830_183018


namespace NUMINAMATH_CALUDE_two_digit_number_representation_l1830_183056

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (num : TwoDigitNumber) : ℕ :=
  10 * num.tens + num.units

theorem two_digit_number_representation (n m : ℕ) (h : n ≥ 1 ∧ n ≤ 9 ∧ m ≤ 9) :
  let num : TwoDigitNumber := ⟨n, m, h⟩
  num.value = 10 * n + m := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_representation_l1830_183056


namespace NUMINAMATH_CALUDE_megan_total_markers_l1830_183042

/-- The number of markers Megan initially had -/
def initial_markers : ℕ := 217

/-- The number of markers Robert gave to Megan -/
def received_markers : ℕ := 109

/-- The total number of markers Megan has -/
def total_markers : ℕ := initial_markers + received_markers

theorem megan_total_markers : total_markers = 326 := by
  sorry

end NUMINAMATH_CALUDE_megan_total_markers_l1830_183042


namespace NUMINAMATH_CALUDE_max_students_before_new_year_l1830_183005

/-- The maximum number of students before New Year given the conditions -/
theorem max_students_before_new_year
  (N : ℕ) -- Total number of students before New Year
  (M : ℕ) -- Number of boys before New Year
  (k : ℕ) -- Percentage of boys before New Year
  (ℓ : ℕ) -- Percentage of boys after New Year
  (h1 : M = k * N / 100) -- Condition relating M, k, and N
  (h2 : ℓ < 100) -- ℓ is less than 100
  (h3 : 100 * (M + 1) = ℓ * (N + 3)) -- Condition after New Year
  : N ≤ 197 := by
  sorry

end NUMINAMATH_CALUDE_max_students_before_new_year_l1830_183005


namespace NUMINAMATH_CALUDE_selling_price_calculation_l1830_183036

def cost_price : ℝ := 1500
def loss_percentage : ℝ := 14.000000000000002

theorem selling_price_calculation (cost_price : ℝ) (loss_percentage : ℝ) :
  let loss_amount := (loss_percentage / 100) * cost_price
  let selling_price := cost_price - loss_amount
  selling_price = 1290 := by sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l1830_183036


namespace NUMINAMATH_CALUDE_smallest_n_with_conditions_l1830_183009

def is_sum_of_identical_digits (n : ℕ) (count : ℕ) : Prop :=
  ∃ (d : ℕ), d ≤ 9 ∧ n = count * d

theorem smallest_n_with_conditions : 
  let n := 6036
  (n > 0) ∧ 
  (n % 2010 = 0) ∧ 
  (n % 2012 = 0) ∧ 
  (n % 2013 = 0) ∧
  (is_sum_of_identical_digits n 2010) ∧
  (is_sum_of_identical_digits n 2012) ∧
  (is_sum_of_identical_digits n 2013) ∧
  (∀ m : ℕ, m > 0 ∧ 
            m % 2010 = 0 ∧ 
            m % 2012 = 0 ∧ 
            m % 2013 = 0 ∧
            is_sum_of_identical_digits m 2010 ∧
            is_sum_of_identical_digits m 2012 ∧
            is_sum_of_identical_digits m 2013 
            → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_conditions_l1830_183009


namespace NUMINAMATH_CALUDE_gap_height_from_wire_extension_l1830_183088

/-- Given a sphere of radius R and a wire wrapped around its equator,
    if the wire's length is increased by L, the resulting gap height h
    between the sphere and the wire is given by h = L / (2π). -/
theorem gap_height_from_wire_extension (R L : ℝ) (h : ℝ) 
    (hR : R > 0) (hL : L > 0) : 
    2 * π * (R + h) = 2 * π * R + L → h = L / (2 * π) := by
  sorry

end NUMINAMATH_CALUDE_gap_height_from_wire_extension_l1830_183088


namespace NUMINAMATH_CALUDE_problem_solution_l1830_183065

theorem problem_solution (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xyz : x * y * z = 1) (h_x_z : x + 1 / z = 7) (h_y_x : y + 1 / x = 31) :
  z + 1 / y = 5 / 27 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1830_183065


namespace NUMINAMATH_CALUDE_sqrt_equation_l1830_183035

theorem sqrt_equation (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_l1830_183035


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1830_183083

theorem fraction_equation_solution : ∃ X : ℝ, (2/5 : ℝ) * (5/9 : ℝ) * X = 0.11111111111111112 ∧ X = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1830_183083


namespace NUMINAMATH_CALUDE_mary_marbles_count_l1830_183075

/-- The number of yellow marbles Mary and Joan have in total -/
def total_marbles : ℕ := 12

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := total_marbles - joan_marbles

theorem mary_marbles_count : mary_marbles = 9 := by
  sorry

end NUMINAMATH_CALUDE_mary_marbles_count_l1830_183075


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l1830_183031

theorem average_of_remaining_numbers
  (n : ℕ)
  (total : ℕ)
  (subset : ℕ)
  (avg_all : ℚ)
  (avg_subset : ℚ)
  (h_total : n = 5)
  (h_subset : subset = 3)
  (h_avg_all : avg_all = 6)
  (h_avg_subset : avg_subset = 4) :
  (n * avg_all - subset * avg_subset) / (n - subset) = 9 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l1830_183031


namespace NUMINAMATH_CALUDE_percentage_calculation_l1830_183037

theorem percentage_calculation (x y z : ℝ) 
  (hx : 0.2 * x = 200)
  (hy : 0.3 * y = 150)
  (hz : 0.4 * z = 80) :
  (0.9 * x - 0.6 * y) + 0.5 * (x + y + z) = 1450 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1830_183037


namespace NUMINAMATH_CALUDE_arithmetic_sequence_slope_l1830_183021

/-- For an arithmetic sequence {a_n} where a_2 - a_4 = 2, 
    the slope of the line containing points (n, a_n) is -1 -/
theorem arithmetic_sequence_slope (a : ℕ → ℝ) (h : a 2 - a 4 = 2) :
  ∃ b : ℝ, ∀ n : ℕ, a n = -n + b := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_slope_l1830_183021


namespace NUMINAMATH_CALUDE_liangliang_speed_l1830_183057

theorem liangliang_speed (initial_distance : ℝ) (remaining_distance : ℝ) (time : ℝ) (mingming_speed : ℝ) :
  initial_distance = 3000 →
  remaining_distance = 2900 →
  time = 20 →
  mingming_speed = 80 →
  ∃ (liangliang_speed : ℝ), (liangliang_speed = 75 ∨ liangliang_speed = 85) ∧
    (initial_distance - remaining_distance = (mingming_speed - liangliang_speed) * time) :=
by sorry

end NUMINAMATH_CALUDE_liangliang_speed_l1830_183057


namespace NUMINAMATH_CALUDE_curve_translation_l1830_183090

-- Define the original curve
def original_curve (x y : ℝ) : Prop :=
  y * Real.cos x + 2 * y - 1 = 0

-- Define the translated curve
def translated_curve (x y : ℝ) : Prop :=
  (y + 1) * Real.sin x + 2 * y + 1 = 0

-- Theorem statement
theorem curve_translation :
  ∀ x y : ℝ, original_curve (x - π/2) (y + 1) ↔ translated_curve x y :=
by sorry

end NUMINAMATH_CALUDE_curve_translation_l1830_183090


namespace NUMINAMATH_CALUDE_coupon_savings_difference_l1830_183084

/-- Represents the savings from a coupon given a price -/
def CouponSavings (price : ℝ) : (ℝ → ℝ) → ℝ := fun coupon => coupon price

/-- Coupon A: 20% off the listed price -/
def CouponA (price : ℝ) : ℝ := 0.2 * price

/-- Coupon B: $30 off the listed price -/
def CouponB (price : ℝ) : ℝ := 30

/-- Coupon C: 20% off the amount exceeding $100 -/
def CouponC (price : ℝ) : ℝ := 0.2 * (price - 100)

/-- The lowest price where Coupon A saves at least as much as Coupon B or C -/
def x : ℝ := 150

/-- The highest price where Coupon A saves at least as much as Coupon B or C -/
def y : ℝ := 300

theorem coupon_savings_difference :
  ∀ price : ℝ, price > 100 →
  (x ≤ price ∧ price ≤ y) ↔
  (CouponSavings price CouponA ≥ CouponSavings price CouponB ∧
   CouponSavings price CouponA ≥ CouponSavings price CouponC) →
  y - x = 150 := by sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_l1830_183084


namespace NUMINAMATH_CALUDE_num_non_congruent_triangles_l1830_183094

/-- Represents a point on a 2D grid --/
structure GridPoint where
  x : ℚ
  y : ℚ

/-- The set of points on the grid --/
def gridPoints : Finset GridPoint := sorry

/-- Predicate to check if three points form a triangle --/
def isTriangle (p q r : GridPoint) : Prop := sorry

/-- Predicate to check if two triangles are congruent --/
def areCongruent (t1 t2 : GridPoint × GridPoint × GridPoint) : Prop := sorry

/-- The set of all possible triangles formed by the grid points --/
def allTriangles : Finset (GridPoint × GridPoint × GridPoint) := sorry

/-- The set of non-congruent triangles --/
def nonCongruentTriangles : Finset (GridPoint × GridPoint × GridPoint) := sorry

theorem num_non_congruent_triangles :
  Finset.card nonCongruentTriangles = 4 := by sorry

end NUMINAMATH_CALUDE_num_non_congruent_triangles_l1830_183094


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1830_183077

theorem rectangle_area_change (original_area : ℝ) (length_decrease : ℝ) (width_increase : ℝ) :
  original_area = 600 →
  length_decrease = 0.2 →
  width_increase = 0.05 →
  original_area * (1 - length_decrease) * (1 + width_increase) = 504 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1830_183077


namespace NUMINAMATH_CALUDE_solve_for_w_l1830_183052

theorem solve_for_w (u v w : ℝ) 
  (eq1 : 10 * u + 8 * v + 5 * w = 160)
  (eq2 : v = u + 3)
  (eq3 : w = 2 * v) : 
  w = 13.5714 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_w_l1830_183052


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l1830_183010

theorem triangle_side_calculation (a b c : ℝ) (A B C : ℝ) : 
  a = 10 → B = 2 * π / 3 → C = π / 6 → 
  A + B + C = π → 
  a / Real.sin A = b / Real.sin B → 
  b = 10 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l1830_183010


namespace NUMINAMATH_CALUDE_certain_number_problem_l1830_183000

theorem certain_number_problem : ∃ x : ℤ, (3005 - 3000 + x = 2705) ∧ (x = 2700) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1830_183000


namespace NUMINAMATH_CALUDE_negation_proposition_true_l1830_183015

theorem negation_proposition_true : ∃ (a b : ℝ), (2 * a + b > 5) ∧ (a ≠ 2 ∨ b ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_proposition_true_l1830_183015


namespace NUMINAMATH_CALUDE_telephone_network_connections_l1830_183082

/-- The number of distinct connections in a network of telephones -/
def distinct_connections (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

/-- Theorem: In a network of 7 telephones, where each telephone is connected to 6 others,
    the total number of distinct connections is 21. -/
theorem telephone_network_connections :
  distinct_connections 7 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_telephone_network_connections_l1830_183082


namespace NUMINAMATH_CALUDE_equal_intercept_line_equations_l1830_183086

/-- A line passing through a point with equal intercepts on both axes --/
structure EqualInterceptLine where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_point : a * 2 + b * 3 + c = 0
  equal_intercepts : a ≠ 0 ∧ b ≠ 0 → -c/a = -c/b

/-- The equations of the line passing through (2,3) with equal intercepts --/
theorem equal_intercept_line_equations :
  ∀ (l : EqualInterceptLine), 
  (l.a = 3 ∧ l.b = -2 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -5) := by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equations_l1830_183086


namespace NUMINAMATH_CALUDE_polynomial_properties_l1830_183062

/-- Definition of our polynomial -/
def p (x y : ℝ) : ℝ := -x^3 - 2*x^2*y^2 + 3*y^2

/-- The number of terms in our polynomial -/
def num_terms : ℕ := 3

/-- The degree of our polynomial -/
def poly_degree : ℕ := 4

/-- Theorem stating the properties of our polynomial -/
theorem polynomial_properties :
  (num_terms = 3) ∧ (poly_degree = 4) := by sorry

end NUMINAMATH_CALUDE_polynomial_properties_l1830_183062


namespace NUMINAMATH_CALUDE_three_digit_sum_theorem_l1830_183099

-- Define a function to represent a three-digit number
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

-- Define a function to represent a two-digit number
def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- Define the condition for the problem
def satisfies_condition (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  three_digit_number a b c = 
    two_digit_number a b + two_digit_number a c +
    two_digit_number b a + two_digit_number b c +
    two_digit_number c a + two_digit_number c b

-- State the theorem
theorem three_digit_sum_theorem :
  ∀ a b c : ℕ, satisfies_condition a b c ↔ 
    (a = 1 ∧ b = 3 ∧ c = 2) ∨ 
    (a = 2 ∧ b = 6 ∧ c = 4) ∨ 
    (a = 3 ∧ b = 9 ∧ c = 6) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_theorem_l1830_183099


namespace NUMINAMATH_CALUDE_square_roots_problem_l1830_183014

theorem square_roots_problem (a : ℝ) (x : ℝ) (h1 : x > 0) 
  (h2 : (3*a + 1)^2 = x) (h3 : (-a - 3)^2 = x) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1830_183014


namespace NUMINAMATH_CALUDE_helen_washing_time_l1830_183063

/-- The time it takes Helen to wash pillowcases each time -/
def washing_time (weeks_between_washes : ℕ) (minutes_per_year : ℕ) (weeks_per_year : ℕ) : ℕ :=
  minutes_per_year / (weeks_per_year / weeks_between_washes)

/-- Theorem stating that Helen's pillowcase washing time is 30 minutes -/
theorem helen_washing_time :
  washing_time 4 390 52 = 30 := by
  sorry

end NUMINAMATH_CALUDE_helen_washing_time_l1830_183063


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l1830_183049

theorem sqrt_sum_equality (a b m n : ℚ) : 
  Real.sqrt a + Real.sqrt b = 1 →
  Real.sqrt a = m + (a - b) / 2 →
  Real.sqrt b = n - (a - b) / 2 →
  m^2 + n^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l1830_183049


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_l1830_183007

theorem cos_squared_minus_sin_squared (α : Real) :
  (∃ (x y : Real), x = 1 ∧ y = 2 ∧ y / x = Real.tan α) →
  Real.cos α ^ 2 - Real.sin α ^ 2 = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_l1830_183007


namespace NUMINAMATH_CALUDE_pizza_portion_eaten_l1830_183058

theorem pizza_portion_eaten (total_slices : ℕ) (slices_left : ℕ) :
  total_slices = 16 → slices_left = 4 →
  (total_slices - slices_left : ℚ) / total_slices = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_portion_eaten_l1830_183058


namespace NUMINAMATH_CALUDE_prescription_final_cost_l1830_183095

/-- Calculates the final cost of a prescription after cash back and rebate --/
def final_cost (original_price : ℝ) (cashback_rate : ℝ) (rebate : ℝ) : ℝ :=
  original_price - (original_price * cashback_rate + rebate)

/-- Theorem stating the final cost of the prescription --/
theorem prescription_final_cost :
  final_cost 150 0.1 25 = 110 := by
  sorry

end NUMINAMATH_CALUDE_prescription_final_cost_l1830_183095


namespace NUMINAMATH_CALUDE_water_polo_team_selection_result_l1830_183034

/-- The number of ways to choose a starting team in water polo -/
def water_polo_team_selection (total_players : ℕ) (team_size : ℕ) (goalie_count : ℕ) : ℕ :=
  Nat.choose total_players goalie_count * Nat.choose (total_players - goalie_count) (team_size - goalie_count)

/-- Theorem: The number of ways to choose a starting team of 9 players (including 2 goalies) from a team of 20 members is 6,046,560 -/
theorem water_polo_team_selection_result :
  water_polo_team_selection 20 9 2 = 6046560 := by
  sorry

end NUMINAMATH_CALUDE_water_polo_team_selection_result_l1830_183034


namespace NUMINAMATH_CALUDE_interest_equality_problem_l1830_183023

theorem interest_equality_problem (total : ℚ) (x : ℚ) : 
  total = 2743 →
  (x * 3 * 8) / 100 = ((total - x) * 5 * 3) / 100 →
  total - x = 1688 := by
  sorry

end NUMINAMATH_CALUDE_interest_equality_problem_l1830_183023


namespace NUMINAMATH_CALUDE_eddy_travel_time_l1830_183013

-- Define the given conditions
def freddy_time : ℝ := 3
def eddy_distance : ℝ := 600
def freddy_distance : ℝ := 300
def speed_ratio : ℝ := 2

-- Define the theorem
theorem eddy_travel_time :
  ∀ (eddy_time : ℝ),
    eddy_time > 0 →
    eddy_distance / eddy_time = speed_ratio * (freddy_distance / freddy_time) →
    eddy_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_eddy_travel_time_l1830_183013


namespace NUMINAMATH_CALUDE_reunion_handshakes_l1830_183059

/-- Calculates the number of handshakes in a group --/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents the reunion scenario --/
structure Reunion :=
  (total_boys : ℕ)
  (left_handed_boys : ℕ)
  (h_left_handed_le_total : left_handed_boys ≤ total_boys)

/-- Calculates the total number of handshakes at the reunion --/
def total_handshakes (r : Reunion) : ℕ :=
  handshakes r.left_handed_boys + handshakes (r.total_boys - r.left_handed_boys)

/-- Theorem stating that the total number of handshakes is 34 for the given scenario --/
theorem reunion_handshakes :
  ∀ (r : Reunion), r.total_boys = 12 → r.left_handed_boys = 4 → total_handshakes r = 34 :=
by
  sorry


end NUMINAMATH_CALUDE_reunion_handshakes_l1830_183059


namespace NUMINAMATH_CALUDE_max_intersection_points_l1830_183016

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  vertices : Fin 4 → ℝ × ℝ

/-- Represents an equilateral triangle in 2D space -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Represents the configuration of a circle, rectangle, and equilateral triangle -/
structure Configuration where
  circle : Circle
  rectangle : Rectangle
  triangle : EquilateralTriangle

/-- Predicate to check if two polygons are distinct -/
def are_distinct (rect : Rectangle) (tri : EquilateralTriangle) : Prop :=
  ∀ (i : Fin 4) (j : Fin 3), rect.vertices i ≠ tri.vertices j

/-- Predicate to check if two polygons do not overlap -/
def do_not_overlap (rect : Rectangle) (tri : EquilateralTriangle) : Prop :=
  sorry  -- Definition of non-overlapping polygons

/-- Function to count the number of intersection points -/
def count_intersections (config : Configuration) : ℕ :=
  sorry  -- Definition to count intersection points

/-- Theorem stating the maximum number of intersection points -/
theorem max_intersection_points (config : Configuration) 
  (h1 : are_distinct config.rectangle config.triangle)
  (h2 : do_not_overlap config.rectangle config.triangle) :
  count_intersections config ≤ 14 :=
sorry

end NUMINAMATH_CALUDE_max_intersection_points_l1830_183016


namespace NUMINAMATH_CALUDE_pant_cost_l1830_183098

theorem pant_cost (num_shirts : ℕ) (shirt_cost : ℕ) (total_cost : ℕ) : 
  num_shirts = 10 →
  shirt_cost = 6 →
  total_cost = 100 →
  ∃ (pant_cost : ℕ), 
    pant_cost = 8 ∧ 
    num_shirts * shirt_cost + (num_shirts / 2) * pant_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_pant_cost_l1830_183098


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1830_183045

theorem fraction_to_decimal (h : 160 = 2^5 * 5) : 7 / 160 = 0.175 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1830_183045


namespace NUMINAMATH_CALUDE_inequality_preservation_l1830_183030

theorem inequality_preservation (m n : ℝ) (h : m > n) : m / 5 > n / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1830_183030


namespace NUMINAMATH_CALUDE_root_ratio_equality_l1830_183054

/-- 
Given a complex polynomial z^4 + az^3 + bz^2 + cz + d with roots p, q, r, s,
if a^2d = c^2 and c ≠ 0, then p/r = s/q.
-/
theorem root_ratio_equality (a b c d p q r s : ℂ) : 
  p * q * r * s = d → 
  p + q + r + s = -a → 
  a^2 * d = c^2 → 
  c ≠ 0 → 
  p / r = s / q := by
sorry

end NUMINAMATH_CALUDE_root_ratio_equality_l1830_183054


namespace NUMINAMATH_CALUDE_largest_c_value_l1830_183017

theorem largest_c_value : ∃ (c_max : ℚ), 
  (∀ c : ℚ, (3 * c + 4) * (c - 2) = 9 * c → c ≤ c_max) ∧ 
  ((3 * c_max + 4) * (c_max - 2) = 9 * c_max) ∧
  c_max = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_c_value_l1830_183017


namespace NUMINAMATH_CALUDE_stamps_per_page_l1830_183039

theorem stamps_per_page (book1 book2 book3 : ℕ) 
  (h1 : book1 = 924) 
  (h2 : book2 = 1386) 
  (h3 : book3 = 1848) : 
  Nat.gcd book1 (Nat.gcd book2 book3) = 462 := by
  sorry

end NUMINAMATH_CALUDE_stamps_per_page_l1830_183039


namespace NUMINAMATH_CALUDE_amy_work_hours_l1830_183024

/-- Calculates the required weekly hours for a given total earnings, number of weeks, and hourly rate -/
def required_weekly_hours (total_earnings : ℚ) (num_weeks : ℚ) (hourly_rate : ℚ) : ℚ :=
  total_earnings / (num_weeks * hourly_rate)

/-- Represents Amy's work scenario -/
theorem amy_work_hours 
  (summer_weekly_hours : ℚ) 
  (summer_weeks : ℚ) 
  (summer_earnings : ℚ) 
  (school_weeks : ℚ) 
  (school_earnings : ℚ)
  (h1 : summer_weekly_hours = 45)
  (h2 : summer_weeks = 8)
  (h3 : summer_earnings = 3600)
  (h4 : school_weeks = 24)
  (h5 : school_earnings = 3600) :
  required_weekly_hours school_earnings school_weeks 
    (summer_earnings / (summer_weekly_hours * summer_weeks)) = 15 := by
  sorry

#eval required_weekly_hours 3600 24 (3600 / (45 * 8))

end NUMINAMATH_CALUDE_amy_work_hours_l1830_183024


namespace NUMINAMATH_CALUDE_scientific_notation_63000_l1830_183061

theorem scientific_notation_63000 : 63000 = 6.3 * (10 ^ 4) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_63000_l1830_183061


namespace NUMINAMATH_CALUDE_power_sum_simplification_l1830_183041

theorem power_sum_simplification :
  -2^2002 + (-1)^2003 + 2^2004 + (-1)^2005 = 3 * 2^2002 - 2 := by sorry

end NUMINAMATH_CALUDE_power_sum_simplification_l1830_183041


namespace NUMINAMATH_CALUDE_sum_of_digits_l1830_183060

theorem sum_of_digits (A B C D E : ℕ) : A + B + C + D + E = 32 :=
  by
  have h1 : A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 := by sorry
  have h2 : 3 * E % 10 = 1 := by sorry
  have h3 : 3 * A + (B + C + D + 2) = 20 := by sorry
  have h4 : B + C + D = 19 := by sorry
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_l1830_183060


namespace NUMINAMATH_CALUDE_monkey_bird_problem_l1830_183040

theorem monkey_bird_problem (initial_monkeys initial_birds : ℕ) 
  (eating_monkeys : ℕ) (percentage_monkeys : ℚ) :
  initial_monkeys = 6 →
  initial_birds = 6 →
  eating_monkeys = 2 →
  percentage_monkeys = 60 / 100 →
  ∃ (birds_eaten : ℕ),
    birds_eaten * eating_monkeys = initial_birds - (initial_monkeys / percentage_monkeys - initial_monkeys) ∧
    birds_eaten = 1 :=
by sorry

end NUMINAMATH_CALUDE_monkey_bird_problem_l1830_183040


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_2_range_of_a_for_x_in_1_to_3_l1830_183050

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - |2*x - 1|

-- Statement for part 1
theorem solution_set_for_a_eq_2 :
  {x : ℝ | f 2 x + 3 ≥ 0} = {x : ℝ | -4 ≤ x ∧ x ≤ 2} := by sorry

-- Statement for part 2
theorem range_of_a_for_x_in_1_to_3 :
  (∀ x ∈ Set.Icc 1 3, f a x ≤ 3) → a ∈ Set.Icc (-3) 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_2_range_of_a_for_x_in_1_to_3_l1830_183050


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1830_183085

theorem inequality_system_solution :
  ∀ x : ℝ,
  (x + 1 > 7 - 2*x ∧ x ≤ (4 + 2*x) / 3) ↔ (2 < x ∧ x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1830_183085


namespace NUMINAMATH_CALUDE_regular_polygon_sides_and_exterior_angle_l1830_183074

/-- 
Theorem: For a regular polygon with n sides, if the sum of its interior angles 
is greater than the sum of its exterior angles by 360°, then n = 6 and each 
exterior angle measures 60°.
-/
theorem regular_polygon_sides_and_exterior_angle (n : ℕ) : 
  (n ≥ 3) →  -- Ensure the polygon has at least 3 sides
  (180 * (n - 2) = 360 + 360) →  -- Sum of interior angles equals 360° + sum of exterior angles
  (n = 6 ∧ 360 / n = 60) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_and_exterior_angle_l1830_183074


namespace NUMINAMATH_CALUDE_cylinder_volume_l1830_183078

/-- Given a cylinder whose lateral surface unfolds into a rectangle with length 2a and width a, 
    its volume is either a³/π or a³/(2π) -/
theorem cylinder_volume (a : ℝ) (h : a > 0) :
  ∃ (V : ℝ), (V = a^3 / π ∨ V = a^3 / (2*π)) ∧
  ∃ (r h : ℝ), r > 0 ∧ h > 0 ∧
  ((2*π*r = 2*a ∧ h = a) ∨ (2*π*r = a ∧ h = 2*a)) ∧
  V = π * r^2 * h :=
sorry

end NUMINAMATH_CALUDE_cylinder_volume_l1830_183078


namespace NUMINAMATH_CALUDE_fraction_value_l1830_183079

theorem fraction_value (a b c d : ℝ) 
  (ha : a = 4 * b) 
  (hb : b = 3 * c) 
  (hc : c = 5 * d) : 
  (a * c) / (b * d) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1830_183079


namespace NUMINAMATH_CALUDE_quadratic_cubic_inequalities_l1830_183071

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 3

theorem quadratic_cubic_inequalities 
  (m n a b : ℝ) 
  (h1 : n = 0)
  (h2 : -2 * m + n = -2)
  (h3 : m * 1^2 + n * 1 = a * 1^3 + b * 1 - 3)
  (h4 : 2 * m * 1 + n = 3 * a * 1^2 + b) :
  ∃ (k p : ℝ), k = 2 ∧ p = -1 ∧ 
  (∀ x > 0, f m n x ≥ k * x + p ∧ g a b x ≤ k * x + p) := by
sorry

end NUMINAMATH_CALUDE_quadratic_cubic_inequalities_l1830_183071


namespace NUMINAMATH_CALUDE_limit_special_function_l1830_183033

/-- The limit of (5 - 4/cos(x))^(1/sin^2(3x)) as x approaches 0 is e^(-2/9) -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
      |(5 - 4 / Real.cos x) ^ (1 / Real.sin (3 * x) ^ 2) - Real.exp (-2/9)| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_special_function_l1830_183033


namespace NUMINAMATH_CALUDE_monthly_snake_feeding_cost_l1830_183038

/-- Proves that the monthly cost per snake is $10, given Harry's pet ownership and feeding costs. -/
theorem monthly_snake_feeding_cost (num_geckos num_iguanas num_snakes : ℕ)
  (gecko_cost iguana_cost : ℚ) (total_annual_cost : ℚ) :
  num_geckos = 3 →
  num_iguanas = 2 →
  num_snakes = 4 →
  gecko_cost = 15 →
  iguana_cost = 5 →
  total_annual_cost = 1140 →
  (num_geckos * gecko_cost + num_iguanas * iguana_cost + num_snakes * 10) * 12 = total_annual_cost :=
by sorry

end NUMINAMATH_CALUDE_monthly_snake_feeding_cost_l1830_183038


namespace NUMINAMATH_CALUDE_parallelogram_base_l1830_183089

/-- Given a parallelogram with area 180 square centimeters and height 10 cm, its base is 18 cm. -/
theorem parallelogram_base (area height base : ℝ) : 
  area = 180 ∧ height = 10 ∧ area = base * height → base = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l1830_183089


namespace NUMINAMATH_CALUDE_calculate_interest_rate_l1830_183028

/-- Calculates the interest rate at which B lends money to C -/
theorem calculate_interest_rate (principal : ℝ) (rate_ab : ℝ) (time : ℝ) (gain : ℝ) : 
  principal = 4000 →
  rate_ab = 10 →
  time = 3 →
  gain = 180 →
  ∃ (rate_bc : ℝ), rate_bc = 11.5 ∧ 
    principal * (rate_bc / 100) * time = principal * (rate_ab / 100) * time + gain :=
by sorry


end NUMINAMATH_CALUDE_calculate_interest_rate_l1830_183028


namespace NUMINAMATH_CALUDE_weight_loss_problem_l1830_183003

/-- Given four people who lost weight, prove that the last two people each lost 28 kg. -/
theorem weight_loss_problem (total_loss weight_loss1 weight_loss2 weight_loss3 weight_loss4 : ℕ) :
  total_loss = 103 →
  weight_loss1 = 27 →
  weight_loss2 = weight_loss1 - 7 →
  weight_loss3 = weight_loss4 →
  total_loss = weight_loss1 + weight_loss2 + weight_loss3 + weight_loss4 →
  weight_loss3 = 28 ∧ weight_loss4 = 28 := by
  sorry


end NUMINAMATH_CALUDE_weight_loss_problem_l1830_183003


namespace NUMINAMATH_CALUDE_reeyas_average_score_l1830_183022

def scores : List ℕ := [55, 67, 76, 82, 55]

theorem reeyas_average_score :
  (scores.sum : ℚ) / scores.length = 67 := by sorry

end NUMINAMATH_CALUDE_reeyas_average_score_l1830_183022


namespace NUMINAMATH_CALUDE_no_snow_probability_l1830_183044

def probability_of_snow : ℚ := 2/3

def days : ℕ := 5

theorem no_snow_probability :
  (1 - probability_of_snow) ^ days = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l1830_183044


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l1830_183053

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_condition : a + b + c = 1)
  (product_sum_condition : a * b + a * c + b * c = -6)
  (product_condition : a * b * c = -3) :
  a^3 + b^3 + c^3 = 27 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l1830_183053


namespace NUMINAMATH_CALUDE_percent_of_500_l1830_183080

theorem percent_of_500 : (110 : ℚ) / 100 * 500 = 550 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_500_l1830_183080


namespace NUMINAMATH_CALUDE_no_solution_iff_m_eq_seven_l1830_183012

theorem no_solution_iff_m_eq_seven (m : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - m) / (x - 8)) ↔ m = 7 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_eq_seven_l1830_183012


namespace NUMINAMATH_CALUDE_min_value_sum_of_fractions_l1830_183046

theorem min_value_sum_of_fractions (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) ≥ 3 * Real.sqrt 3 / 2 ∧
  (x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) = 3 * Real.sqrt 3 / 2 ↔ 
   x = Real.sqrt 3 / 3 ∧ y = Real.sqrt 3 / 3 ∧ z = Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_fractions_l1830_183046


namespace NUMINAMATH_CALUDE_order_of_abc_l1830_183068

theorem order_of_abc (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ineq : a^2 + b^2 < a^2 + c^2 ∧ a^2 + c^2 < b^2 + c^2) : 
  a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l1830_183068


namespace NUMINAMATH_CALUDE_angle_less_iff_sin_less_l1830_183043

theorem angle_less_iff_sin_less (A B : Real) (hA : 0 < A) (hB : B < π) (hAB : A + B < π) :
  A < B ↔ Real.sin A < Real.sin B := by sorry

end NUMINAMATH_CALUDE_angle_less_iff_sin_less_l1830_183043


namespace NUMINAMATH_CALUDE_equation_solutions_l1830_183067

-- Define the equation
def equation (x : ℝ) : Prop := x / 50 = Real.cos x

-- State the theorem
theorem equation_solutions :
  ∃! (s : Finset ℝ), s.card = 31 ∧ ∀ x, x ∈ s ↔ equation x :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1830_183067
