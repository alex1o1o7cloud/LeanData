import Mathlib

namespace tire_usage_proof_l2963_296313

-- Define the number of tires
def total_tires : ℕ := 6

-- Define the total miles traveled by the car
def total_miles : ℕ := 45000

-- Define the number of tires used at any given time
def tires_in_use : ℕ := 4

-- Define the function to calculate miles per tire
def miles_per_tire (total_tires : ℕ) (total_miles : ℕ) (tires_in_use : ℕ) : ℕ :=
  (total_miles * tires_in_use) / total_tires

-- Theorem statement
theorem tire_usage_proof :
  miles_per_tire total_tires total_miles tires_in_use = 30000 := by
  sorry

end tire_usage_proof_l2963_296313


namespace arccos_one_half_l2963_296370

theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := by sorry

end arccos_one_half_l2963_296370


namespace percentage_equality_l2963_296342

theorem percentage_equality (x : ℝ) (h : 0.3 * 0.4 * x = 24) : 0.4 * 0.3 * x = 24 := by
  sorry

end percentage_equality_l2963_296342


namespace equal_ratios_fraction_l2963_296395

theorem equal_ratios_fraction (x y z : ℝ) (h : x/2 = y/3 ∧ y/3 = z/4) :
  (x + y) / (3*y - 2*z) = 5 := by
  sorry

end equal_ratios_fraction_l2963_296395


namespace sqrt_product_equals_sqrt_of_product_l2963_296302

theorem sqrt_product_equals_sqrt_of_product :
  Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_equals_sqrt_of_product_l2963_296302


namespace sum_of_n_values_l2963_296310

theorem sum_of_n_values (n : ℤ) : 
  (∃ (S : Finset ℤ), (∀ m ∈ S, (∃ k : ℤ, 24 = k * (2 * m - 1))) ∧ 
   (∀ m : ℤ, (∃ k : ℤ, 24 = k * (2 * m - 1)) → m ∈ S) ∧ 
   (Finset.sum S id = 3)) := by
sorry

end sum_of_n_values_l2963_296310


namespace businessmen_neither_coffee_nor_tea_l2963_296386

def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 13
def both_drinkers : ℕ := 7

theorem businessmen_neither_coffee_nor_tea :
  total_businessmen - (coffee_drinkers + tea_drinkers - both_drinkers) = 9 :=
by sorry

end businessmen_neither_coffee_nor_tea_l2963_296386


namespace coin_problem_l2963_296331

theorem coin_problem (x y : ℕ) : 
  x + y = 40 →  -- Total number of coins
  2 * x + 5 * y = 125 →  -- Total amount of money
  y = 15  -- Number of 5-dollar coins
  := by sorry

end coin_problem_l2963_296331


namespace negation_equivalence_l2963_296348

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 2*x = 0) ↔ (∀ x : ℝ, x^2 - 2*x ≠ 0) := by sorry

end negation_equivalence_l2963_296348


namespace exists_in_set_A_l2963_296350

/-- Sum of digits of a positive integer -/
def digit_sum (n : ℕ+) : ℕ := sorry

/-- Predicate for a number having all non-zero digits -/
def all_digits_nonzero (n : ℕ+) : Prop := sorry

/-- Number of digits in a positive integer -/
def num_digits (n : ℕ+) : ℕ := sorry

/-- The main theorem -/
theorem exists_in_set_A (k : ℕ+) : 
  ∃ x : ℕ+, num_digits x = k ∧ all_digits_nonzero x ∧ (digit_sum x ∣ x) := by
  sorry

end exists_in_set_A_l2963_296350


namespace six_cows_satisfy_condition_unique_cow_count_l2963_296322

/-- Represents the farm with cows and chickens -/
structure Farm where
  cows : ℕ
  chickens : ℕ

/-- The total number of legs on the farm -/
def totalLegs (f : Farm) : ℕ := 5 * f.cows + 2 * f.chickens

/-- The total number of heads on the farm -/
def totalHeads (f : Farm) : ℕ := f.cows + f.chickens

/-- The farm satisfies the given condition -/
def satisfiesCondition (f : Farm) : Prop :=
  totalLegs f = 20 + 2 * totalHeads f

/-- Theorem stating that the farm with 6 cows satisfies the condition -/
theorem six_cows_satisfy_condition :
  ∃ (f : Farm), f.cows = 6 ∧ satisfiesCondition f :=
sorry

/-- Theorem stating that 6 is the only number of cows that satisfies the condition -/
theorem unique_cow_count :
  ∀ (f : Farm), satisfiesCondition f → f.cows = 6 :=
sorry

end six_cows_satisfy_condition_unique_cow_count_l2963_296322


namespace arithmetic_square_root_of_64_l2963_296307

theorem arithmetic_square_root_of_64 : Real.sqrt 64 = 8 := by sorry

end arithmetic_square_root_of_64_l2963_296307


namespace circles_centers_form_rectangle_l2963_296308

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

def inside (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 < (c2.radius - c1.radius)^2

def rectangle (a b c d : ℝ × ℝ) : Prop :=
  let ab := (b.1 - a.1, b.2 - a.2)
  let bc := (c.1 - b.1, c.2 - b.2)
  let cd := (d.1 - c.1, d.2 - c.2)
  let da := (a.1 - d.1, a.2 - d.2)
  ab.1 * bc.1 + ab.2 * bc.2 = 0 ∧
  bc.1 * cd.1 + bc.2 * cd.2 = 0 ∧
  cd.1 * da.1 + cd.2 * da.2 = 0 ∧
  da.1 * ab.1 + da.2 * ab.2 = 0 ∧
  ab.1^2 + ab.2^2 = cd.1^2 + cd.2^2 ∧
  bc.1^2 + bc.2^2 = da.1^2 + da.2^2

theorem circles_centers_form_rectangle 
  (C C1 C2 C3 C4 : Circle)
  (h1 : C.radius = 2)
  (h2 : C1.radius = 1)
  (h3 : C2.radius = 1)
  (h4 : tangent C1 C2)
  (h5 : inside C1 C)
  (h6 : inside C2 C)
  (h7 : inside C3 C)
  (h8 : inside C4 C)
  (h9 : tangent C3 C)
  (h10 : tangent C3 C1)
  (h11 : tangent C3 C2)
  (h12 : tangent C4 C)
  (h13 : tangent C4 C1)
  (h14 : tangent C4 C3)
  : rectangle C.center C1.center C3.center C4.center :=
sorry

end circles_centers_form_rectangle_l2963_296308


namespace negation_of_exists_ellipse_eccentricity_lt_one_l2963_296345

/-- An ellipse is a geometric shape with an eccentricity. -/
structure Ellipse where
  eccentricity : ℝ

/-- The negation of "There exists an ellipse with an eccentricity e < 1" 
    is equivalent to "The eccentricity e ≥ 1 for any ellipse". -/
theorem negation_of_exists_ellipse_eccentricity_lt_one :
  (¬ ∃ (e : Ellipse), e.eccentricity < 1) ↔ (∀ (e : Ellipse), e.eccentricity ≥ 1) :=
sorry

end negation_of_exists_ellipse_eccentricity_lt_one_l2963_296345


namespace investment_quoted_price_l2963_296366

/-- Calculates the quoted price of shares given investment details -/
def quoted_price (total_investment : ℚ) (nominal_value : ℚ) (dividend_rate : ℚ) (annual_income : ℚ) : ℚ :=
  let dividend_per_share := (dividend_rate / 100) * nominal_value
  let number_of_shares := annual_income / dividend_per_share
  total_investment / number_of_shares

/-- Theorem stating that given the investment details, the quoted price is 9.5 -/
theorem investment_quoted_price :
  quoted_price 4940 10 14 728 = 9.5 := by
  sorry

end investment_quoted_price_l2963_296366


namespace odd_sum_probability_redesigned_board_l2963_296300

/-- Represents the redesigned dartboard -/
structure Dartboard where
  outer_radius : ℝ
  inner_radius : ℝ
  inner_points : Fin 3 → ℕ
  outer_points : Fin 3 → ℕ

/-- The probability of getting an odd sum when throwing two darts -/
def odd_sum_probability (d : Dartboard) : ℝ :=
  sorry

/-- The redesigned dartboard as described in the problem -/
def redesigned_board : Dartboard :=
  { outer_radius := 8
    inner_radius := 4
    inner_points := λ i => if i = 0 then 3 else 1
    outer_points := λ i => if i = 0 then 2 else 3 }

/-- Theorem stating the probability of an odd sum on the redesigned board -/
theorem odd_sum_probability_redesigned_board :
    odd_sum_probability redesigned_board = 4 / 9 :=
  sorry

end odd_sum_probability_redesigned_board_l2963_296300


namespace f_2015_value_l2963_296341

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2015_value (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_periodic : ∀ x, f (x + 4) = f x)
  (h_f_1 : f 1 = 2) : 
  f 2015 = -2 := by
  sorry

end f_2015_value_l2963_296341


namespace consecutive_hits_theorem_l2963_296399

/-- The number of ways to arrange 8 shots with 3 hits, where exactly 2 hits are consecutive -/
def consecutive_hits_arrangements (total_shots : ℕ) (total_hits : ℕ) (consecutive_hits : ℕ) : ℕ :=
  if total_shots = 8 ∧ total_hits = 3 ∧ consecutive_hits = 2 then
    30
  else
    0

/-- Theorem stating that the number of arrangements is 30 -/
theorem consecutive_hits_theorem :
  consecutive_hits_arrangements 8 3 2 = 30 := by
  sorry

end consecutive_hits_theorem_l2963_296399


namespace minimum_implies_a_range_l2963_296359

/-- A function f with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + a

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

/-- Theorem stating that if f has a minimum in (0,2), then a is in (0,4) -/
theorem minimum_implies_a_range (a : ℝ) :
  (∃ x₀ ∈ Set.Ioo 0 2, ∀ x ∈ Set.Ioo 0 2, f a x₀ ≤ f a x) →
  a ∈ Set.Ioo 0 4 :=
sorry

end minimum_implies_a_range_l2963_296359


namespace other_factors_of_twenty_l2963_296351

theorem other_factors_of_twenty (y : ℕ) : 
  y = 20 ∧ y % 5 = 0 ∧ y % 8 ≠ 0 → 
  (∀ x : ℕ, x ≠ 1 ∧ x ≠ 5 ∧ y % x = 0 → x = 2 ∨ x = 4 ∨ x = 10) :=
by sorry

end other_factors_of_twenty_l2963_296351


namespace brown_eyes_fraction_l2963_296306

theorem brown_eyes_fraction (total_students : ℕ) 
  (brown_eyes_black_hair : ℕ) 
  (h1 : total_students = 18) 
  (h2 : brown_eyes_black_hair = 6) 
  (h3 : brown_eyes_black_hair * 2 = brown_eyes_black_hair + brown_eyes_black_hair) :
  (brown_eyes_black_hair * 2 : ℚ) / total_students = 2 / 3 := by
  sorry

end brown_eyes_fraction_l2963_296306


namespace base_seven_528_l2963_296343

def base_seven_representation (n : ℕ) : List ℕ :=
  sorry

theorem base_seven_528 :
  base_seven_representation 528 = [1, 3, 5, 3] := by
  sorry

end base_seven_528_l2963_296343


namespace lena_time_to_counter_l2963_296335

/-- The time it takes Lena to reach the counter given her initial movement and remaining distance -/
theorem lena_time_to_counter (initial_distance : ℝ) (initial_time : ℝ) (remaining_distance_meters : ℝ) :
  initial_distance = 40 →
  initial_time = 20 →
  remaining_distance_meters = 100 →
  (remaining_distance_meters * 3.28084) / (initial_distance / initial_time) = 164.042 := by
sorry

end lena_time_to_counter_l2963_296335


namespace fund_price_calculation_l2963_296397

theorem fund_price_calculation (initial_price : ℝ) 
  (monday_change tuesday_change wednesday_change thursday_change friday_change : ℝ) :
  initial_price = 35 →
  monday_change = 4.5 →
  tuesday_change = 4 →
  wednesday_change = -1 →
  thursday_change = -2.5 →
  friday_change = -6 →
  initial_price + monday_change + tuesday_change + wednesday_change + thursday_change + friday_change = 34 := by
  sorry

end fund_price_calculation_l2963_296397


namespace valid_spy_placement_exists_l2963_296332

/-- Represents a position on the board -/
structure Position where
  x : Fin 6
  y : Fin 6

/-- Represents the vision of a spy -/
inductive Vision where
  | ahead : Position → Vision
  | right : Position → Vision
  | left : Position → Vision

/-- Checks if a spy at the given position can see the target position -/
def canSee (spyPos : Position) (targetPos : Position) : Prop :=
  ∃ (v : Vision),
    match v with
    | Vision.ahead p => p.x = spyPos.x ∧ p.y = spyPos.y + 1 ∨ p.y = spyPos.y + 2
    | Vision.right p => p.x = spyPos.x + 1 ∧ p.y = spyPos.y
    | Vision.left p => p.x = spyPos.x - 1 ∧ p.y = spyPos.y

/-- A valid spy placement is a list of 18 positions where no spy can see another -/
def ValidSpyPlacement (placement : List Position) : Prop :=
  placement.length = 18 ∧
  ∀ (spy1 spy2 : Position),
    spy1 ∈ placement → spy2 ∈ placement → spy1 ≠ spy2 →
    ¬(canSee spy1 spy2 ∨ canSee spy2 spy1)

/-- Theorem stating that a valid spy placement exists -/
theorem valid_spy_placement_exists : ∃ (placement : List Position), ValidSpyPlacement placement :=
  sorry

end valid_spy_placement_exists_l2963_296332


namespace triangle_translation_inconsistency_l2963_296344

structure Point where
  x : ℝ
  y : ℝ

def Triangle (A B C : Point) := Unit

def isTranslation (A B C A' B' C' : Point) : Prop :=
  ∃ dx dy : ℝ, 
    A'.x = A.x + dx ∧ A'.y = A.y + dy ∧
    B'.x = B.x + dx ∧ B'.y = B.y + dy ∧
    C'.x = C.x + dx ∧ C'.y = C.y + dy

def correctYCoordinates (A B C A' B' C' : Point) : Prop :=
  A'.y = A.y - 3 ∧ B'.y = B.y - 3 ∧ C'.y = C.y - 3

def oneCorrectXCoordinate (A B C A' B' C' : Point) : Prop :=
  (A'.x = A.x ∧ B'.x ≠ B.x - 1 ∧ C'.x ≠ C.x + 1) ∨
  (A'.x ≠ A.x ∧ B'.x = B.x - 1 ∧ C'.x ≠ C.x + 1) ∨
  (A'.x ≠ A.x ∧ B'.x ≠ B.x - 1 ∧ C'.x = C.x + 1)

theorem triangle_translation_inconsistency 
  (A B C A' B' C' : Point)
  (h1 : A = ⟨0, 3⟩)
  (h2 : B = ⟨-1, 0⟩)
  (h3 : C = ⟨1, 0⟩)
  (h4 : A' = ⟨0, 0⟩)
  (h5 : B' = ⟨-2, -3⟩)
  (h6 : C' = ⟨2, -3⟩)
  (h7 : correctYCoordinates A B C A' B' C')
  (h8 : oneCorrectXCoordinate A B C A' B' C') :
  ¬(isTranslation A B C A' B' C') ∧
  ((A' = ⟨0, 0⟩ ∧ B' = ⟨-1, -3⟩ ∧ C' = ⟨1, -3⟩) ∨
   (A' = ⟨-1, 0⟩ ∧ B' = ⟨-2, -3⟩ ∧ C' = ⟨0, -3⟩) ∨
   (A' = ⟨1, 0⟩ ∧ B' = ⟨0, -3⟩ ∧ C' = ⟨2, -3⟩)) :=
by
  sorry

end triangle_translation_inconsistency_l2963_296344


namespace ellipse_eccentricity_special_case_l2963_296327

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a circle with radius r -/
structure Circle where
  r : ℝ
  h_positive : 0 < r

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Eccentricity of ellipse under specific conditions -/
theorem ellipse_eccentricity_special_case 
  (E : Ellipse) 
  (O : Circle)
  (B : Point)
  (A : Point)
  (h_circle : O.r = E.a)
  (h_B_on_y : B.x = 0 ∧ B.y = E.a)
  (h_B_on_ellipse : B.x^2 / E.a^2 + B.y^2 / E.b^2 = 1)
  (h_B_on_circle : B.x^2 + B.y^2 = O.r^2)
  (h_A_on_circle : A.x^2 + A.y^2 = O.r^2)
  (h_tangent : ∃ (m : ℝ), (A.y - B.y) = m * (A.x - B.x) ∧ 
               ∀ (x y : ℝ), y = m * (x - B.x) + B.y → x^2 / E.a^2 + y^2 / E.b^2 ≥ 1)
  (h_angle : Real.cos (60 * π / 180) = (A.x * B.x + A.y * B.y) / (O.r^2)) :
  let e := Real.sqrt (E.a^2 - E.b^2) / E.a
  e = Real.sqrt 3 / 3 := by
    sorry

end ellipse_eccentricity_special_case_l2963_296327


namespace factorization_of_2x_squared_minus_10_l2963_296303

theorem factorization_of_2x_squared_minus_10 :
  ∀ x : ℝ, 2 * x^2 - 10 = 2 * (x + Real.sqrt 5) * (x - Real.sqrt 5) := by
  sorry

end factorization_of_2x_squared_minus_10_l2963_296303


namespace coin_combinations_theorem_l2963_296379

/-- Represents the denominations of coins available in kopecks -/
def coin_denominations : List Nat := [1, 2, 5, 10, 20, 50]

/-- Represents the total amount to be made in kopecks -/
def total_amount : Nat := 100

/-- 
  Calculates the number of ways to make the total amount using the given coin denominations
  
  @param coins The list of available coin denominations
  @param amount The total amount to be made
  @return The number of ways to make the total amount
-/
def count_ways (coins : List Nat) (amount : Nat) : Nat :=
  sorry

theorem coin_combinations_theorem : 
  count_ways coin_denominations total_amount = 4562 := by
  sorry

end coin_combinations_theorem_l2963_296379


namespace rectangle_area_difference_l2963_296339

theorem rectangle_area_difference (x : ℝ) : 
  (2 * (x + 7)) * (2 * (x + 5)) - (3 * (2 * x - 3)) * (3 * (x - 2)) = -14 * x^2 + 111 * x + 86 := by
  sorry

end rectangle_area_difference_l2963_296339


namespace square_product_inequality_l2963_296369

theorem square_product_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 > a*b ∧ a*b > b^2 := by sorry

end square_product_inequality_l2963_296369


namespace parabola_properties_l2963_296340

/-- Parabola properties -/
structure Parabola where
  a : ℝ
  x₁ : ℝ
  x₂ : ℝ
  y : ℝ → ℝ
  h₁ : a ≠ 0
  h₂ : x₁ ≠ x₂
  h₃ : ∀ x, y x = x^2 + (1 - 2*a)*x + a^2
  h₄ : y x₁ = 0
  h₅ : y x₂ = 0

/-- Main theorem about the parabola -/
theorem parabola_properties (p : Parabola) :
  (0 < p.a ∧ p.a < 1/4 ∧ p.x₁ < 0 ∧ p.x₂ < 0) ∧
  (p.y 0 - 2 = -p.x₁ - p.x₂ → p.a = -3) :=
sorry

end parabola_properties_l2963_296340


namespace committee_formation_count_l2963_296312

/-- Represents a department in the science division -/
inductive Department : Type
| Biology : Department
| Physics : Department
| Chemistry : Department
| Mathematics : Department

/-- Represents the gender of a professor -/
inductive Gender : Type
| Male : Gender
| Female : Gender

/-- Represents a professor with their department and gender -/
structure Professor :=
  (dept : Department)
  (gender : Gender)

/-- The total number of departments -/
def total_departments : Nat := 4

/-- The number of male professors in each department -/
def male_professors_per_dept : Nat := 3

/-- The number of female professors in each department -/
def female_professors_per_dept : Nat := 3

/-- The total number of professors in the committee -/
def committee_size : Nat := 8

/-- The number of male professors required in the committee -/
def required_males : Nat := 4

/-- The number of female professors required in the committee -/
def required_females : Nat := 4

/-- The number of departments that should contribute exactly 2 professors -/
def depts_with_two_profs : Nat := 2

/-- The minimum number of professors required from the Mathematics department -/
def min_math_profs : Nat := 2

/-- Calculates the number of ways to form the committee -/
def count_committee_formations : Nat :=
  sorry

/-- Theorem stating that the number of ways to form the committee is 1944 -/
theorem committee_formation_count :
  count_committee_formations = 1944 :=
sorry

end committee_formation_count_l2963_296312


namespace gcd_of_N_is_12_l2963_296384

def N (a b c d : ℕ) : ℤ :=
  (a - b) * (c - d) * (a - c) * (b - d) * (a - d) * (b - c)

theorem gcd_of_N_is_12 :
  ∃ (k : ℕ), ∀ (a b c d : ℕ), 
    (∃ (n : ℤ), N a b c d = 12 * n) ∧
    (∀ (m : ℕ), m > 12 → ¬(∃ (l : ℤ), N a b c d = m * l)) :=
sorry

end gcd_of_N_is_12_l2963_296384


namespace length_of_A_l2963_296321

def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 7)

def on_line_y_eq_x (p : ℝ × ℝ) : Prop := p.1 = p.2

def intersect_at (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    (p.1 + t * (r.1 - p.1) = q.1) ∧
    (p.2 + t * (r.2 - p.2) = q.2)

theorem length_of_A'B' :
  ∃ A' B' : ℝ × ℝ, 
    on_line_y_eq_x A' ∧
    on_line_y_eq_x B' ∧
    intersect_at A A' C ∧
    intersect_at B B' C ∧
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 10 * Real.sqrt 2 / 11 := by
  sorry

end length_of_A_l2963_296321


namespace regular_octagon_area_l2963_296346

/-- The area of a regular octagon with side length 2√2 is 16 + 16√2 -/
theorem regular_octagon_area : 
  let s : ℝ := 2 * Real.sqrt 2
  8 * (s^2 / (4 * Real.tan (π/8))) = 16 + 16 * Real.sqrt 2 := by
  sorry

end regular_octagon_area_l2963_296346


namespace cost_price_of_ball_l2963_296316

/-- 
Given:
- The selling price of 13 balls is 720 Rs.
- The loss incurred is equal to the cost price of 5 balls.

Prove that the cost price of one ball is 90 Rs.
-/
theorem cost_price_of_ball (selling_price : ℕ) (num_balls : ℕ) (loss_balls : ℕ) :
  selling_price = 720 →
  num_balls = 13 →
  loss_balls = 5 →
  ∃ (cost_price : ℕ), 
    cost_price * num_balls - selling_price = cost_price * loss_balls ∧
    cost_price = 90 :=
by sorry

end cost_price_of_ball_l2963_296316


namespace square_area_from_adjacent_points_l2963_296388

/-- Given two adjacent points (1,2) and (4,6) on a square, the area of the square is 25 -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (4, 6)
  let distance_squared := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2
  let area := distance_squared
  area = 25 := by sorry

end square_area_from_adjacent_points_l2963_296388


namespace p_false_q_true_l2963_296309

theorem p_false_q_true (h1 : p ∨ q) (h2 : ¬p) : ¬p ∧ q := by
  sorry

end p_false_q_true_l2963_296309


namespace sum_three_digit_even_integers_eq_247050_l2963_296364

/-- The sum of all three-digit positive even integers -/
def sum_three_digit_even_integers : ℕ :=
  let first : ℕ := 100  -- First three-digit even integer
  let last : ℕ := 998   -- Last three-digit even integer
  let count : ℕ := (last - first) / 2 + 1  -- Number of terms
  count * (first + last) / 2

/-- Theorem stating that the sum of all three-digit positive even integers is 247050 -/
theorem sum_three_digit_even_integers_eq_247050 :
  sum_three_digit_even_integers = 247050 := by
  sorry

#eval sum_three_digit_even_integers

end sum_three_digit_even_integers_eq_247050_l2963_296364


namespace solve_candy_store_problem_l2963_296390

def candy_store_problem (initial_money : ℚ) (gum_packs : ℕ) (gum_price : ℚ) 
  (chocolate_bars : ℕ) (candy_canes : ℕ) (candy_cane_price : ℚ) (money_left : ℚ) : Prop :=
  ∃ (chocolate_bar_price : ℚ),
    initial_money = 
      gum_packs * gum_price + 
      chocolate_bars * chocolate_bar_price + 
      candy_canes * candy_cane_price + 
      money_left ∧
    chocolate_bar_price = 1

theorem solve_candy_store_problem :
  candy_store_problem 10 3 1 5 2 (1/2) 1 := by
  sorry

end solve_candy_store_problem_l2963_296390


namespace cube_root_equation_solution_l2963_296376

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 2) ^ (1/3 : ℝ) = 2 :=
by
  sorry

end cube_root_equation_solution_l2963_296376


namespace cyclist_average_speed_l2963_296396

/-- Calculates the average speed of a cyclist who drives four laps of equal distance
    at different speeds. -/
theorem cyclist_average_speed (d : ℝ) (h : d > 0) :
  let speeds := [6, 12, 18, 24]
  let total_distance := 4 * d
  let total_time := d / 6 + d / 12 + d / 18 + d / 24
  total_distance / total_time = 288 / 25 := by
sorry

#eval (288 : ℚ) / 25  -- To verify the result is approximately 11.52

end cyclist_average_speed_l2963_296396


namespace stick_swap_triangle_formation_l2963_296314

/-- Represents a set of three stick lengths -/
structure StickSet where
  s₁ : ℝ
  s₂ : ℝ
  s₃ : ℝ
  sum_is_one : s₁ + s₂ + s₃ = 1
  all_positive : 0 < s₁ ∧ 0 < s₂ ∧ 0 < s₃

/-- Checks if a triangle can be formed from the given stick lengths -/
def can_form_triangle (s : StickSet) : Prop :=
  s.s₁ < s.s₂ + s.s₃ ∧ s.s₂ < s.s₁ + s.s₃ ∧ s.s₃ < s.s₁ + s.s₂

theorem stick_swap_triangle_formation 
  (v_initial w_initial : StickSet)
  (v_can_form_initial : can_form_triangle v_initial)
  (w_can_form_initial : can_form_triangle w_initial)
  (v_final w_final : StickSet)
  (swap_occurred : ∃ (i j : Fin 3), 
    v_final.s₁ + v_final.s₂ + v_final.s₃ + w_final.s₁ + w_final.s₂ + w_final.s₃ = 
    v_initial.s₁ + v_initial.s₂ + v_initial.s₃ + w_initial.s₁ + w_initial.s₂ + w_initial.s₃)
  (v_cannot_form_final : ¬can_form_triangle v_final) :
  can_form_triangle w_final :=
sorry

end stick_swap_triangle_formation_l2963_296314


namespace tenth_term_of_specific_sequence_l2963_296323

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r ^ (n - 1)

theorem tenth_term_of_specific_sequence :
  geometric_sequence 5 (3/4) 10 = 98415/262144 := by
  sorry

end tenth_term_of_specific_sequence_l2963_296323


namespace product_def_l2963_296382

theorem product_def (a b c d e f : ℝ) 
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : (a * f) / (c * d) = 0.75) :
  d * e * f = 250 := by
sorry

end product_def_l2963_296382


namespace point_coordinates_l2963_296393

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of a 2D coordinate system -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: Given the conditions, prove that the point P has coordinates (-2, 5) -/
theorem point_coordinates (P : Point) 
  (h1 : SecondQuadrant P) 
  (h2 : DistanceToXAxis P = 5) 
  (h3 : DistanceToYAxis P = 2) : 
  P.x = -2 ∧ P.y = 5 := by
  sorry

end point_coordinates_l2963_296393


namespace bake_sale_group_l2963_296372

theorem bake_sale_group (p : ℕ) : 
  p > 0 → 
  (p : ℚ) / 2 - 2 = (2 * p : ℚ) / 5 → 
  (p : ℚ) / 2 = 10 := by
  sorry

end bake_sale_group_l2963_296372


namespace condition_for_cubic_equation_l2963_296304

theorem condition_for_cubic_equation (a b : ℝ) (h : a * b ≠ 0) :
  (a - b = 1) ↔ (a^3 - b^3 - a*b - a^2 - b^2 = 0) :=
by sorry

end condition_for_cubic_equation_l2963_296304


namespace least_sum_of_exponents_for_800_l2963_296398

def isPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def sumOfDistinctPowersOfTwo (n : ℕ) (powers : List ℕ) : Prop :=
  (powers.map (λ k => 2^k)).sum = n ∧ powers.Nodup

theorem least_sum_of_exponents_for_800 :
  ∀ (powers : List ℕ),
    sumOfDistinctPowersOfTwo 800 powers →
    powers.length ≥ 3 →
    powers.sum ≥ 22 :=
by sorry

end least_sum_of_exponents_for_800_l2963_296398


namespace complementary_event_correct_l2963_296334

/-- A bag containing red and white balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- An event in the sample space of drawing balls -/
inductive Event
  | AtLeastOneWhite
  | AllRed

/-- The complementary event function -/
def complementary_event : Event → Event
  | Event.AtLeastOneWhite => Event.AllRed
  | Event.AllRed => Event.AtLeastOneWhite

theorem complementary_event_correct (bag : Bag) (h1 : bag.red = 3) (h2 : bag.white = 2) :
  complementary_event Event.AtLeastOneWhite = Event.AllRed :=
by sorry

end complementary_event_correct_l2963_296334


namespace smallest_pencil_collection_l2963_296315

theorem smallest_pencil_collection (P : ℕ) : 
  P > 2 ∧ 
  P % 5 = 2 ∧ 
  P % 9 = 2 ∧ 
  P % 11 = 2 → 
  (∀ Q : ℕ, Q > 2 ∧ Q % 5 = 2 ∧ Q % 9 = 2 ∧ Q % 11 = 2 → P ≤ Q) →
  P = 497 := by
sorry

end smallest_pencil_collection_l2963_296315


namespace amusement_park_cost_per_trip_l2963_296365

/-- Calculates the cost per trip to an amusement park given the number of sons,
    cost per pass, and number of trips by each son. -/
def cost_per_trip (num_sons : ℕ) (cost_per_pass : ℚ) (trips_oldest : ℕ) (trips_youngest : ℕ) : ℚ :=
  (num_sons * cost_per_pass) / (trips_oldest + trips_youngest)

/-- Theorem stating that for the given inputs, the cost per trip is $4.00 -/
theorem amusement_park_cost_per_trip :
  cost_per_trip 2 100 35 15 = 4 := by
  sorry

end amusement_park_cost_per_trip_l2963_296365


namespace cara_age_l2963_296385

/-- Given the age relationships in Cara's family, prove Cara's age --/
theorem cara_age :
  ∀ (cara_age mom_age grandma_age : ℕ),
    cara_age = mom_age - 20 →
    mom_age = grandma_age - 15 →
    grandma_age = 75 →
    cara_age = 40 := by
  sorry

end cara_age_l2963_296385


namespace smallest_gcd_multiple_l2963_296381

theorem smallest_gcd_multiple (p q : ℕ+) (h : Nat.gcd p q = 15) :
  (∀ p' q' : ℕ+, Nat.gcd p' q' = 15 → Nat.gcd (8 * p') (18 * q') ≥ 30) ∧
  (∃ p' q' : ℕ+, Nat.gcd p' q' = 15 ∧ Nat.gcd (8 * p') (18 * q') = 30) :=
by sorry

end smallest_gcd_multiple_l2963_296381


namespace min_income_2020_l2963_296325

/-- Represents the per capita income growth over a 40-year period -/
def income_growth (initial : ℝ) (mid : ℝ) (final : ℝ) : Prop :=
  ∃ (x : ℝ), 
    initial * (1 + x)^20 ≥ mid ∧
    initial * (1 + x)^40 ≥ final

/-- Theorem stating the minimum per capita income in 2020 based on 1980 and 2000 data -/
theorem min_income_2020 : income_growth 250 800 2560 := by
  sorry

end min_income_2020_l2963_296325


namespace right_triangle_angles_l2963_296355

-- Define a right triangle
structure RightTriangle where
  a : ℝ  -- length of one leg
  b : ℝ  -- length of the other leg
  h : ℝ  -- length of the hypotenuse
  right_angle : a^2 + b^2 = h^2  -- Pythagorean theorem

-- Define the quadrilateral formed by the perpendicular bisector
structure Quadrilateral where
  d1 : ℝ  -- length of one diagonal
  d2 : ℝ  -- length of the other diagonal

-- Main theorem
theorem right_triangle_angles (triangle : RightTriangle) (quad : Quadrilateral) :
  quad.d1 / quad.d2 = (1 + Real.sqrt 3) / (2 * Real.sqrt 2) →
  ∃ (angle1 angle2 : ℝ),
    angle1 = 15 * π / 180 ∧
    angle2 = 75 * π / 180 ∧
    angle1 + angle2 = π / 2 :=
by sorry

end right_triangle_angles_l2963_296355


namespace not_perfect_square_l2963_296320

theorem not_perfect_square (t : ℤ) : ¬ ∃ k : ℤ, 7 * t + 3 = k ^ 2 := by
  sorry

end not_perfect_square_l2963_296320


namespace largest_three_digit_divisible_by_digits_l2963_296336

def is_divisible_by_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0 → n % d = 0

theorem largest_three_digit_divisible_by_digits :
  ∀ n : ℕ, 800 ≤ n → n ≤ 899 → is_divisible_by_digits n →
  n ≤ 864 :=
sorry

end largest_three_digit_divisible_by_digits_l2963_296336


namespace unique_color_for_X_l2963_296311

/-- Represents the four colors used in the grid --/
inductive Color
| Red
| Blue
| Yellow
| Green

/-- Represents a position in the grid --/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Represents the grid --/
def Grid := Position → Option Color

/-- Checks if two positions are adjacent (share a vertex) --/
def adjacent (p1 p2 : Position) : Prop :=
  (p1.x = p2.x ∧ (p1.y = p2.y + 1 ∨ p1.y = p2.y - 1)) ∨
  (p1.y = p2.y ∧ (p1.x = p2.x + 1 ∨ p1.x = p2.x - 1)) ∨
  (p1.x = p2.x + 1 ∧ p1.y = p2.y + 1) ∨
  (p1.x = p2.x - 1 ∧ p1.y = p2.y - 1) ∨
  (p1.x = p2.x + 1 ∧ p1.y = p2.y - 1) ∨
  (p1.x = p2.x - 1 ∧ p1.y = p2.y + 1)

/-- Checks if the grid coloring is valid --/
def valid_coloring (g : Grid) : Prop :=
  ∀ p1 p2 : Position, adjacent p1 p2 →
    (g p1).isSome ∧ (g p2).isSome →
    (g p1 ≠ g p2)

/-- The position of cell X --/
def X : Position := ⟨5, 5⟩

/-- Theorem: There exists a unique color for cell X in a valid 4-color grid --/
theorem unique_color_for_X (g : Grid) (h : valid_coloring g) :
  ∃! c : Color, g X = some c :=
sorry

end unique_color_for_X_l2963_296311


namespace planes_parallel_if_line_perpendicular_to_both_l2963_296319

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_line_perpendicular_to_both 
  (a : Line) (α β : Plane) (h1 : α ≠ β) :
  perp a α → perp a β → para α β :=
sorry

end planes_parallel_if_line_perpendicular_to_both_l2963_296319


namespace range_of_a_l2963_296330

/-- Proposition p: For all x in [1,2], x^2 - a ≥ 0 -/
def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

/-- Proposition q: The equation x^2 + 2ax + a + 2 = 0 has solutions -/
def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + a + 2 = 0

/-- If propositions p and q are both true, then a ∈ (-∞, -1] -/
theorem range_of_a (a : ℝ) (h_p : prop_p a) (h_q : prop_q a) : a ∈ Set.Iic (-1) :=
sorry

end range_of_a_l2963_296330


namespace arithmetic_sequence_problem_l2963_296338

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 4th term is 23 and the 6th term is 51,
    the 8th term is 79. -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
    (h_arith : ArithmeticSequence a) 
    (h_4th : a 4 = 23)
    (h_6th : a 6 = 51) : 
    a 8 = 79 := by
  sorry


end arithmetic_sequence_problem_l2963_296338


namespace book_collection_problem_l2963_296360

theorem book_collection_problem (shared_books books_alice books_bob_unique : ℕ) 
  (h1 : shared_books = 12)
  (h2 : books_alice = 26)
  (h3 : books_bob_unique = 8) :
  books_alice - shared_books + books_bob_unique = 22 := by
  sorry

end book_collection_problem_l2963_296360


namespace intersection_A_complement_B_l2963_296389

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {2, 3, 4}

-- Define set B
def B : Set Nat := {1, 2}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {3, 4} := by
  sorry

end intersection_A_complement_B_l2963_296389


namespace mean_score_all_students_l2963_296328

theorem mean_score_all_students
  (score_first : ℝ)
  (score_second : ℝ)
  (ratio_first_to_second : ℚ)
  (h1 : score_first = 90)
  (h2 : score_second = 75)
  (h3 : ratio_first_to_second = 2 / 3) :
  let total_students := (ratio_first_to_second + 1) * students_second
  let total_score := score_first * ratio_first_to_second * students_second + score_second * students_second
  total_score / total_students = 81 :=
by sorry

end mean_score_all_students_l2963_296328


namespace shm_first_return_time_l2963_296352

/-- Time for a particle in Simple Harmonic Motion to first return to origin -/
theorem shm_first_return_time (m k : ℝ) (hm : m > 0) (hk : k > 0) :
  ∃ (t : ℝ), t = π * Real.sqrt (m / k) ∧ t > 0 := by
  sorry

end shm_first_return_time_l2963_296352


namespace quadratic_equation_range_l2963_296357

/-- Given a quadratic equation (m-3)x^2 + 4x + 1 = 0 with real solutions,
    the range of values for m is m ≤ 7 and m ≠ 3 -/
theorem quadratic_equation_range (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) := by
  sorry

end quadratic_equation_range_l2963_296357


namespace cube_roots_sum_power_nine_l2963_296373

/-- Given complex numbers x and y as defined, prove that x⁹ + y⁹ ≠ -1 --/
theorem cube_roots_sum_power_nine (x y : ℂ) : 
  x = (-1 + Complex.I * Real.sqrt 3) / 2 →
  y = (-1 - Complex.I * Real.sqrt 3) / 2 →
  x^9 + y^9 ≠ -1 := by
  sorry

end cube_roots_sum_power_nine_l2963_296373


namespace find_x_l2963_296377

def binary_op (n : ℤ) (x : ℚ) : ℚ := n - (n * x)

theorem find_x : 
  (∀ n : ℤ, n > 3 → binary_op n x ≥ 14) ∧
  (binary_op 3 x < 14) →
  x = -3 := by sorry

end find_x_l2963_296377


namespace northern_car_speed_l2963_296347

/-- Proves that given the initial conditions of two cars and their movement,
    the speed of the northern car must be 80 mph. -/
theorem northern_car_speed 
  (initial_distance : ℝ) 
  (southern_speed : ℝ) 
  (time : ℝ) 
  (final_distance : ℝ) 
  (h1 : initial_distance = 300) 
  (h2 : southern_speed = 60) 
  (h3 : time = 5) 
  (h4 : final_distance = 500) : 
  ∃ v : ℝ, v = 80 ∧ 
  final_distance^2 = initial_distance^2 + (time * v)^2 + (time * southern_speed)^2 :=
by
  sorry


end northern_car_speed_l2963_296347


namespace smoothie_servings_calculation_l2963_296387

/-- Calculates the number of smoothie servings that can be made given the volumes of ingredients and serving size. -/
def smoothie_servings (watermelon_puree : ℕ) (cream : ℕ) (serving_size : ℕ) : ℕ :=
  (watermelon_puree + cream) / serving_size

/-- Theorem: Given 500 ml of watermelon puree, 100 ml of cream, and a serving size of 150 ml, 4 servings of smoothie can be made. -/
theorem smoothie_servings_calculation :
  smoothie_servings 500 100 150 = 4 := by
  sorry

end smoothie_servings_calculation_l2963_296387


namespace peanuts_in_box_l2963_296318

/-- Calculate the final number of peanuts in a box after removing and adding some. -/
def final_peanuts (initial : ℕ) (removed : ℕ) (added : ℕ) : ℕ :=
  initial - removed + added

/-- Theorem stating that for the given values, the final number of peanuts is 13. -/
theorem peanuts_in_box : final_peanuts 4 3 12 = 13 := by
  sorry

end peanuts_in_box_l2963_296318


namespace intersection_equals_zero_one_l2963_296394

def A : Set ℕ := {0, 1, 2, 3}

def B : Set ℕ := {x | ∃ n ∈ A, x = n^2}

def P : Set ℕ := A ∩ B

theorem intersection_equals_zero_one : P = {0, 1} := by sorry

end intersection_equals_zero_one_l2963_296394


namespace ellipse_k_range_l2963_296378

/-- Given an ellipse with equation x^2 / (3-k) + y^2 / (1+k) = 1 and foci on the x-axis,
    the range of k values is (-1, 1) -/
theorem ellipse_k_range (k : ℝ) :
  (∀ x y : ℝ, x^2 / (3-k) + y^2 / (1+k) = 1) →
  (∃ c : ℝ, c > 0 ∧ c^2 = (3-k) - (1+k)) →
  -1 < k ∧ k < 1 :=
by sorry

end ellipse_k_range_l2963_296378


namespace divisibility_by_nine_l2963_296392

theorem divisibility_by_nine (A : ℕ) (h : A < 10) : 
  (7000 + 200 + 10 * A + 4) % 9 = 0 ↔ A = 5 := by sorry

end divisibility_by_nine_l2963_296392


namespace grace_weeding_hours_l2963_296358

/-- Represents Grace's landscaping business earnings in September --/
def graces_earnings (mowing_rate : ℕ) (weeding_rate : ℕ) (mulching_rate : ℕ)
                    (mowing_hours : ℕ) (weeding_hours : ℕ) (mulching_hours : ℕ) : ℕ :=
  mowing_rate * mowing_hours + weeding_rate * weeding_hours + mulching_rate * mulching_hours

/-- Theorem stating that Grace spent 9 hours pulling weeds in September --/
theorem grace_weeding_hours :
  ∀ (mowing_rate weeding_rate mulching_rate mowing_hours mulching_hours total_earnings : ℕ),
    mowing_rate = 6 →
    weeding_rate = 11 →
    mulching_rate = 9 →
    mowing_hours = 63 →
    mulching_hours = 10 →
    total_earnings = 567 →
    ∃ (weeding_hours : ℕ),
      graces_earnings mowing_rate weeding_rate mulching_rate mowing_hours weeding_hours mulching_hours = total_earnings ∧
      weeding_hours = 9 :=
by
  sorry

end grace_weeding_hours_l2963_296358


namespace no_solution_equation_l2963_296368

theorem no_solution_equation : ∀ (a b : ℤ), a^4 + 6 ≠ b^3 := by
  sorry

end no_solution_equation_l2963_296368


namespace photo_ratio_proof_l2963_296354

def claire_photos : ℕ := 6
def robert_photos (claire : ℕ) : ℕ := claire + 12

theorem photo_ratio_proof (lisa : ℕ) (h1 : lisa = robert_photos claire_photos) :
  lisa / claire_photos = 3 := by
  sorry

end photo_ratio_proof_l2963_296354


namespace integral_symmetric_function_l2963_296326

theorem integral_symmetric_function (a : ℝ) (h : a > 0) :
  ∫ x in -a..a, (x^2 * Real.cos x + Real.exp x) / (Real.exp x + 1) = a := by sorry

end integral_symmetric_function_l2963_296326


namespace tan_15_degrees_l2963_296317

theorem tan_15_degrees : Real.tan (15 * π / 180) = 2 - Real.sqrt 3 := by
  sorry

end tan_15_degrees_l2963_296317


namespace sin_n_equals_cos_390_l2963_296353

theorem sin_n_equals_cos_390 (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) :
  Real.sin (n * Real.pi / 180) = Real.cos (390 * Real.pi / 180) → n = 60 := by
sorry

end sin_n_equals_cos_390_l2963_296353


namespace circle_diameter_l2963_296305

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end circle_diameter_l2963_296305


namespace geometric_progression_fourth_term_l2963_296324

theorem geometric_progression_fourth_term :
  ∀ x : ℝ,
  (∃ r : ℝ, r ≠ 0 ∧ (2*x + 2) = r * x ∧ (3*x + 3) = r * (2*x + 2)) →
  ∃ fourth_term : ℝ, fourth_term = -13/2 ∧ (3*x + 3) * r = fourth_term :=
by sorry

end geometric_progression_fourth_term_l2963_296324


namespace existence_of_xy_l2963_296337

theorem existence_of_xy (a b c : ℝ) 
  (h1 : |a| > 2) 
  (h2 : a^2 + b^2 + c^2 = a*b*c + 4) : 
  ∃ x y : ℝ, 
    a = x + 1/x ∧ 
    b = y + 1/y ∧ 
    c = x*y + 1/(x*y) := by
  sorry

end existence_of_xy_l2963_296337


namespace prob_same_color_is_49_128_l2963_296349

-- Define the number of balls of each color
def green_balls : ℕ := 8
def red_balls : ℕ := 5
def blue_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := green_balls + red_balls + blue_balls

-- Define the probability of drawing two balls of the same color
def prob_same_color : ℚ := 
  (green_balls * green_balls + red_balls * red_balls + blue_balls * blue_balls) / 
  (total_balls * total_balls)

-- Theorem statement
theorem prob_same_color_is_49_128 : prob_same_color = 49 / 128 := by
  sorry

end prob_same_color_is_49_128_l2963_296349


namespace inequality_proofs_l2963_296361

def M : Set ℝ := {x | x ≥ 2}

theorem inequality_proofs :
  (∀ (a b c d : ℝ), a ∈ M → b ∈ M → c ∈ M → d ∈ M →
    Real.sqrt (a * b) + Real.sqrt (c * d) ≥ Real.sqrt (a + b) + Real.sqrt (c + d)) ∧
  (∀ (a b c d : ℝ), a ∈ M → b ∈ M → c ∈ M → d ∈ M →
    Real.sqrt (a * b) + Real.sqrt (c * d) ≥ Real.sqrt (a + c) + Real.sqrt (b + d)) :=
by sorry

end inequality_proofs_l2963_296361


namespace a_neg_two_sufficient_not_necessary_l2963_296391

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

def z (a : ℝ) : ℂ := Complex.mk (a^2 - 4) (a + 1)

theorem a_neg_two_sufficient_not_necessary :
  (∃ (a : ℝ), a ≠ -2 ∧ is_pure_imaginary (z a)) ∧
  (∀ (a : ℝ), a = -2 → is_pure_imaginary (z a)) :=
sorry

end a_neg_two_sufficient_not_necessary_l2963_296391


namespace min_copy_paste_actions_l2963_296367

theorem min_copy_paste_actions (n : ℕ) : (2^n - 1 ≥ 1000) ∧ (∀ m : ℕ, m < n → 2^m - 1 < 1000) ↔ n = 10 := by
  sorry

end min_copy_paste_actions_l2963_296367


namespace triangle_side_range_l2963_296362

theorem triangle_side_range (A B C : ℝ) (AB BC : ℝ) :
  AB = Real.sqrt 3 →
  C = π / 3 →
  (∃ (A₁ A₂ : ℝ), A₁ ≠ A₂ ∧ 
    Real.sin A₁ = BC / 2 ∧ 
    Real.sin A₂ = BC / 2 ∧ 
    A₁ ∈ Set.Ioo (π / 3) (2 * π / 3) ∧ 
    A₂ ∈ Set.Ioo (π / 3) (2 * π / 3)) →
  BC > Real.sqrt 3 ∧ BC < 2 := by
sorry


end triangle_side_range_l2963_296362


namespace linear_function_proof_l2963_296375

theorem linear_function_proof (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x + y) = f x + f y) -- linearity
  (h2 : ∀ x y : ℝ, x < y → f x < f y) -- monotonically increasing
  (h3 : ∀ x : ℝ, f (f x) = 16 * x + 9) : -- given condition
  ∀ x : ℝ, f x = 4 * x + 9/5 := by
sorry

end linear_function_proof_l2963_296375


namespace computer_store_discount_rate_l2963_296329

/-- Proves that the discount rate of the second store is approximately 0.87% given the conditions of the problem -/
theorem computer_store_discount_rate (price1 : ℝ) (discount1 : ℝ) (price2 : ℝ) (price_diff : ℝ) :
  price1 = 950 →
  discount1 = 0.06 →
  price2 = 920 →
  price_diff = 19 →
  let discounted_price1 := price1 * (1 - discount1)
  let discounted_price2 := discounted_price1 + price_diff
  let discount2 := (price2 - discounted_price2) / price2
  ∃ ε > 0, |discount2 - 0.0087| < ε :=
by
  sorry

end computer_store_discount_rate_l2963_296329


namespace even_digits_base7_528_l2963_296356

/-- Converts a natural number from base 10 to base 7 --/
def toBase7 (n : ℕ) : List ℕ := sorry

/-- Counts the number of even digits in a list of natural numbers --/
def countEvenDigits (digits : List ℕ) : ℕ := sorry

/-- The number of even digits in the base-7 representation of 528 is 0 --/
theorem even_digits_base7_528 :
  countEvenDigits (toBase7 528) = 0 := by sorry

end even_digits_base7_528_l2963_296356


namespace coefficient_x_squared_is_46_l2963_296374

/-- The coefficient of x^2 in the expansion of (2x^3 + 4x^2 - 3x + 5)(3x^2 - 9x + 1) -/
def coefficient_x_squared : ℤ :=
  let p1 := [2, 4, -3, 5]  -- Coefficients of 2x^3 + 4x^2 - 3x + 5
  let p2 := [3, -9, 1]     -- Coefficients of 3x^2 - 9x + 1
  46

/-- Proof that the coefficient of x^2 in the expansion of (2x^3 + 4x^2 - 3x + 5)(3x^2 - 9x + 1) is 46 -/
theorem coefficient_x_squared_is_46 : coefficient_x_squared = 46 := by
  sorry

end coefficient_x_squared_is_46_l2963_296374


namespace root_sum_cube_product_l2963_296383

theorem root_sum_cube_product (α β : ℝ) : 
  α^2 - 2*α - 4 = 0 → β^2 - 2*β - 4 = 0 → α ≠ β → α^3 + 8*β + 6 = 30 := by
  sorry

end root_sum_cube_product_l2963_296383


namespace v_2002_equals_1_l2963_296333

/-- The function g as defined in the problem --/
def g : ℕ → ℕ
| 1 => 2
| 2 => 3
| 3 => 1
| 4 => 4
| 5 => 5
| _ => 0  -- default case for inputs not in the table

/-- The sequence v defined recursively --/
def v : ℕ → ℕ
| 0 => 2
| (n + 1) => g (v n)

/-- Theorem stating that the 2002nd term of the sequence is 1 --/
theorem v_2002_equals_1 : v 2002 = 1 := by
  sorry


end v_2002_equals_1_l2963_296333


namespace problem_statement_l2963_296371

theorem problem_statement (n : ℕ+) : 
  3 * (Nat.choose (n - 1) (n - 5)) = 5 * (Nat.factorial (n - 2) / Nat.factorial (n - 4)) → n = 9 := by
  sorry

end problem_statement_l2963_296371


namespace linear_function_not_in_third_quadrant_l2963_296363

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 2*x - 3 = 0

-- Define the roots of the quadratic equation
def roots (a b : ℝ) : Prop := quadratic_eq a ∧ quadratic_eq b ∧ a ≠ b

-- Define the linear function
def linear_function (x : ℝ) (a b : ℝ) : ℝ := (a*b - 1)*x + a + b

-- Theorem: The linear function does not pass through the third quadrant
theorem linear_function_not_in_third_quadrant (a b : ℝ) :
  roots a b →
  ∀ x y : ℝ, y = linear_function x a b →
  ¬(x < 0 ∧ y < 0) :=
sorry

end linear_function_not_in_third_quadrant_l2963_296363


namespace intersecting_digit_is_three_l2963_296380

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def powers_of_three : Set ℕ := {n | ∃ m : ℕ, n = 3^m ∧ is_three_digit n}
def powers_of_seven : Set ℕ := {n | ∃ m : ℕ, n = 7^m ∧ is_three_digit n}

theorem intersecting_digit_is_three :
  ∃! d : ℕ, d < 10 ∧ 
  (∃ n ∈ powers_of_three, ∃ i : ℕ, n / 10^i % 10 = d) ∧
  (∃ n ∈ powers_of_seven, ∃ i : ℕ, n / 10^i % 10 = d) :=
by sorry

end intersecting_digit_is_three_l2963_296380


namespace max_value_theorem_l2963_296301

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 - x*y + y^2 = 8) :
  (∃ (z : ℝ), z = x^2 + x*y + y^2 ∧ z ≤ 24) ∧
  (∃ (a b c d : ℕ+), 24 = (a + b * Real.sqrt c) / d ∧ a + b + c + d = 26) :=
by sorry

end max_value_theorem_l2963_296301
