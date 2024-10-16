import Mathlib

namespace NUMINAMATH_CALUDE_undecided_voters_percentage_l2271_227183

theorem undecided_voters_percentage
  (total_polled : ℕ)
  (biff_percentage : ℚ)
  (marty_voters : ℕ)
  (h1 : total_polled = 200)
  (h2 : biff_percentage = 45 / 100)
  (h3 : marty_voters = 94) :
  (total_polled - (marty_voters + (biff_percentage * total_polled).floor)) / total_polled = 8 / 100 := by
  sorry

end NUMINAMATH_CALUDE_undecided_voters_percentage_l2271_227183


namespace NUMINAMATH_CALUDE_sum_of_valid_numbers_mod_1000_l2271_227170

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  (∀ d, d ∈ n.digits 10 → d ≠ 0) ∧
  n % 99 = 0 ∧
  150 % (n / 100) = 0 ∧
  168 % (n % 100) = 0

def sum_of_valid_numbers : ℕ := sorry

theorem sum_of_valid_numbers_mod_1000 :
  sum_of_valid_numbers % 1000 = 108 := by sorry

end NUMINAMATH_CALUDE_sum_of_valid_numbers_mod_1000_l2271_227170


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2271_227156

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 + 24 * x - 4 * y^2 + 16 * y + 44 = 0

/-- The distance between vertices of the hyperbola -/
def vertex_distance : ℝ := 2

/-- Theorem stating that the distance between vertices of the given hyperbola is 2 -/
theorem hyperbola_vertex_distance :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola_equation x₁ y₁ ∧
    hyperbola_equation x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    ∀ (x y : ℝ), hyperbola_equation x y → 
      (x - x₁)^2 + (y - y₁)^2 ≤ (x₁ - x₂)^2 + (y₁ - y₂)^2 ∧
      (x - x₂)^2 + (y - y₂)^2 ≤ (x₁ - x₂)^2 + (y₁ - y₂)^2 ∧
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = vertex_distance^2 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2271_227156


namespace NUMINAMATH_CALUDE_least_n_without_square_l2271_227102

theorem least_n_without_square : ∃ N : ℕ, 
  (∀ k : ℕ, k ≥ N → 
    ∀ i : ℕ, i ∈ Finset.range 1000 → 
      ¬∃ m : ℕ, m^2 = 1000*k + i) ∧
  (∀ k : ℕ, k < N → 
    ∃ i : ℕ, i ∈ Finset.range 1000 ∧ 
      ∃ m : ℕ, m^2 = 1000*k + i) ∧
  N = 282 :=
sorry

end NUMINAMATH_CALUDE_least_n_without_square_l2271_227102


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l2271_227197

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x ≤ 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem 1: A ∪ B = {x | 2 < x < 10}
theorem union_A_B : A ∪ B = {x | 2 < x ∧ x < 10} := by sorry

-- Theorem 2: (Cᵤ A) ∩ B = {x | 2 < x ≤ 3 or 7 < x < 10}
theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = {x | (2 < x ∧ x ≤ 3) ∨ (7 < x ∧ x < 10)} := by sorry

-- Theorem 3: If A ∩ C ≠ ∅, then a ≥ 3
theorem intersection_A_C_nonempty (a : ℝ) : A ∩ C a ≠ ∅ → a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l2271_227197


namespace NUMINAMATH_CALUDE_alex_shirt_count_l2271_227110

/-- Given that:
  - Alex has some new shirts
  - Joe has 3 more new shirts than Alex
  - Ben has a certain number of new shirts more than Joe
  - Ben has 15 new shirts
Prove that Alex has 12 new shirts. -/
theorem alex_shirt_count :
  ∀ (alex_shirts joe_shirts ben_shirts : ℕ),
  joe_shirts = alex_shirts + 3 →
  ben_shirts > joe_shirts →
  ben_shirts = 15 →
  alex_shirts = 12 := by
sorry

end NUMINAMATH_CALUDE_alex_shirt_count_l2271_227110


namespace NUMINAMATH_CALUDE_lopez_family_seating_arrangements_l2271_227150

/-- Represents a family member -/
inductive FamilyMember
| MrLopez
| MrsLopez
| Child1
| Child2
| Child3

/-- Represents a seat in the car -/
inductive Seat
| Driver
| FrontPassenger
| BackLeft
| BackMiddle
| BackRight

/-- A seating arrangement is a function from Seat to FamilyMember -/
def SeatingArrangement := Seat → FamilyMember

/-- Checks if a seating arrangement is valid -/
def isValidArrangement (arr : SeatingArrangement) : Prop :=
  (arr Seat.Driver = FamilyMember.MrLopez ∨ arr Seat.Driver = FamilyMember.MrsLopez) ∧
  (∀ s₁ s₂, s₁ ≠ s₂ → arr s₁ ≠ arr s₂)

/-- The number of valid seating arrangements -/
def numValidArrangements : ℕ := sorry

theorem lopez_family_seating_arrangements :
  numValidArrangements = 48 := by sorry

end NUMINAMATH_CALUDE_lopez_family_seating_arrangements_l2271_227150


namespace NUMINAMATH_CALUDE_saras_quarters_l2271_227136

theorem saras_quarters (initial final given : ℕ) 
  (h1 : initial = 21)
  (h2 : final = 70)
  (h3 : given = final - initial) : 
  given = 49 := by sorry

end NUMINAMATH_CALUDE_saras_quarters_l2271_227136


namespace NUMINAMATH_CALUDE_personal_income_tax_l2271_227115

/-- Personal income tax calculation -/
theorem personal_income_tax (salary : ℕ) (tax_free : ℕ) (rate1 : ℚ) (rate2 : ℚ) (threshold : ℕ) : 
  salary = 2900 ∧ 
  tax_free = 2000 ∧ 
  rate1 = 5/100 ∧ 
  rate2 = 10/100 ∧ 
  threshold = 500 → 
  (min threshold (salary - tax_free) : ℚ) * rate1 + 
  (max 0 ((salary - tax_free) - threshold) : ℚ) * rate2 = 65 := by
sorry

end NUMINAMATH_CALUDE_personal_income_tax_l2271_227115


namespace NUMINAMATH_CALUDE_count_missed_toddlers_l2271_227174

/-- The number of toddlers Bill missed -/
def toddlers_missed (total_count : ℕ) (double_counted : ℕ) (actual_toddlers : ℕ) : ℕ :=
  actual_toddlers - (total_count - double_counted)

/-- Theorem stating that the number of toddlers Bill missed is equal to
    the actual number of toddlers minus the number he actually counted -/
theorem count_missed_toddlers 
  (total_count : ℕ) (double_counted : ℕ) (actual_toddlers : ℕ) :
  toddlers_missed total_count double_counted actual_toddlers = 
  actual_toddlers - (total_count - double_counted) :=
by
  sorry

end NUMINAMATH_CALUDE_count_missed_toddlers_l2271_227174


namespace NUMINAMATH_CALUDE_square_side_equals_two_pi_l2271_227190

/-- The side length of a square whose perimeter is equal to the circumference of a circle with radius 4 units is equal to 2π. -/
theorem square_side_equals_two_pi :
  ∃ x : ℝ, x > 0 ∧ 4 * x = 2 * π * 4 → x = 2 * π :=
by sorry

end NUMINAMATH_CALUDE_square_side_equals_two_pi_l2271_227190


namespace NUMINAMATH_CALUDE_sine_inequality_l2271_227111

theorem sine_inequality (x : Real) (h : 0 < x ∧ x < Real.pi / 4) :
  Real.sin (Real.sin x) < Real.sin x ∧ Real.sin x < Real.sin (Real.tan x) := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l2271_227111


namespace NUMINAMATH_CALUDE_absolute_value_and_exponentiation_l2271_227140

theorem absolute_value_and_exponentiation :
  (abs (-2023) = 2023) ∧ ((-1 : ℤ)^2023 = -1) := by sorry

end NUMINAMATH_CALUDE_absolute_value_and_exponentiation_l2271_227140


namespace NUMINAMATH_CALUDE_power_of_power_l2271_227193

theorem power_of_power (a : ℝ) : (a^3)^3 = a^9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2271_227193


namespace NUMINAMATH_CALUDE_savanna_safari_snake_ratio_l2271_227120

-- Define the number of animals in Safari National Park
def safari_lions : ℕ := 100
def safari_snakes : ℕ := safari_lions / 2
def safari_giraffes : ℕ := safari_snakes - 10

-- Define the number of animals in Savanna National Park
def savanna_lions : ℕ := 2 * safari_lions
def savanna_giraffes : ℕ := safari_giraffes + 20

-- Define the total number of animals in Savanna National Park
def savanna_total : ℕ := 410

-- Define the ratio of snakes in Savanna to Safari
def snake_ratio : ℚ := 3

theorem savanna_safari_snake_ratio :
  snake_ratio = (savanna_total - savanna_lions - savanna_giraffes) / safari_snakes := by
  sorry

end NUMINAMATH_CALUDE_savanna_safari_snake_ratio_l2271_227120


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l2271_227109

theorem triangle_side_lengths :
  ∀ x y z : ℕ,
    x ≥ y ∧ y ≥ z →
    x + y + z = 240 →
    3 * x - 2 * (y + z) = 5 * z + 10 →
    ((x = 113 ∧ y = 112 ∧ z = 15) ∨
     (x = 114 ∧ y = 110 ∧ z = 16) ∨
     (x = 115 ∧ y = 108 ∧ z = 17) ∨
     (x = 116 ∧ y = 106 ∧ z = 18) ∨
     (x = 117 ∧ y = 104 ∧ z = 19) ∨
     (x = 118 ∧ y = 102 ∧ z = 20) ∨
     (x = 119 ∧ y = 100 ∧ z = 21)) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l2271_227109


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2271_227163

/-- Definition of the repeating decimal 0.4444... -/
def repeating_4 : ℚ := 4 / 9

/-- Definition of the repeating decimal 0.3535... -/
def repeating_35 : ℚ := 35 / 99

/-- The sum of the repeating decimals 0.4444... and 0.3535... is equal to 79/99 -/
theorem sum_of_repeating_decimals : repeating_4 + repeating_35 = 79 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2271_227163


namespace NUMINAMATH_CALUDE_function_properties_l2271_227123

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

theorem function_properties (a : ℝ) :
  (∀ x, x < -1 ∨ x > 3 → ∀ y, y > x → f y a < f x a) ∧
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f y a ≤ f x a) ∧
  (f 2 a = 20) →
  a = -2 ∧
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f x a ≤ f y a ∧ f x a = -7) :=
by sorry

#check function_properties

end NUMINAMATH_CALUDE_function_properties_l2271_227123


namespace NUMINAMATH_CALUDE_stock_selection_probabilities_l2271_227121

/-- The number of stocks available for purchase -/
def num_stocks : ℕ := 10

/-- The number of people buying stocks -/
def num_people : ℕ := 3

/-- The probability of all people selecting the same stock -/
def prob_all_same : ℚ := 1 / 100

/-- The probability of at least two people selecting the same stock -/
def prob_at_least_two_same : ℚ := 7 / 25

/-- Theorem stating the probabilities for the stock selection problem -/
theorem stock_selection_probabilities :
  (prob_all_same = 1 / num_stocks ^ (num_people - 1)) ∧
  (prob_at_least_two_same = 
    (1 / num_stocks ^ (num_people - 1)) + 
    (num_stocks * (num_people.choose 2) * (1 / num_stocks ^ 2) * ((num_stocks - 1) / num_stocks))) :=
by sorry

end NUMINAMATH_CALUDE_stock_selection_probabilities_l2271_227121


namespace NUMINAMATH_CALUDE_angle_x_is_72_degrees_l2271_227199

-- Define a regular pentagon
structure RegularPentagon where
  -- All sides are equal (implied by regularity)
  -- All angles are equal (implied by regularity)

-- Define the enclosing structure
structure EnclosingStructure where
  pentagon : RegularPentagon
  -- Squares and triangles enclose the pentagon (implied by the structure)

-- Define the angle x formed by two squares and the pentagon
def angle_x (e : EnclosingStructure) : ℝ := sorry

-- Theorem statement
theorem angle_x_is_72_degrees (e : EnclosingStructure) : 
  angle_x e = 72 := by sorry

end NUMINAMATH_CALUDE_angle_x_is_72_degrees_l2271_227199


namespace NUMINAMATH_CALUDE_granger_cisco_spots_l2271_227187

/-- The number of spots Rover has -/
def rover_spots : ℕ := 46

/-- The number of spots Cisco has -/
def cisco_spots : ℕ := rover_spots / 2 - 5

/-- The number of spots Granger has -/
def granger_spots : ℕ := 5 * cisco_spots

/-- The total number of spots Granger and Cisco have combined -/
def total_spots : ℕ := granger_spots + cisco_spots

theorem granger_cisco_spots : total_spots = 108 := by
  sorry

end NUMINAMATH_CALUDE_granger_cisco_spots_l2271_227187


namespace NUMINAMATH_CALUDE_parabola_coeffs_sum_l2271_227164

/-- Parabola coefficients -/
structure ParabolaCoeffs where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Parabola equation -/
def parabola_equation (coeffs : ParabolaCoeffs) (y : ℝ) : ℝ :=
  coeffs.a * y^2 + coeffs.b * y + coeffs.c

/-- Theorem: Parabola coefficients and their sum -/
theorem parabola_coeffs_sum :
  ∀ (coeffs : ParabolaCoeffs),
  parabola_equation coeffs 5 = 6 ∧
  parabola_equation coeffs 3 = 0 ∧
  (∀ y : ℝ, parabola_equation coeffs y = coeffs.a * (y - 3)^2) →
  coeffs.a = 3/2 ∧ coeffs.b = -9 ∧ coeffs.c = 27/2 ∧
  coeffs.a + coeffs.b + coeffs.c = 6 :=
by sorry


end NUMINAMATH_CALUDE_parabola_coeffs_sum_l2271_227164


namespace NUMINAMATH_CALUDE_no_130_consecutive_numbers_with_900_divisors_l2271_227103

theorem no_130_consecutive_numbers_with_900_divisors :
  ¬ ∃ (n : ℕ), ∀ (k : ℕ), k ∈ Finset.range 130 →
    (Nat.divisors (n + k)).card = 900 :=
sorry

end NUMINAMATH_CALUDE_no_130_consecutive_numbers_with_900_divisors_l2271_227103


namespace NUMINAMATH_CALUDE_all_points_same_number_l2271_227160

-- Define a type for points in the plane
structure Point := (x : ℝ) (y : ℝ)

-- Define a function that assigns a real number to each point
def assign : Point → ℝ := sorry

-- Define a predicate for the inscribed circle property
def inscribedCircleProperty (assign : Point → ℝ) : Prop :=
  ∀ A B C : Point,
  ∃ I : Point,
  assign I = (assign A + assign B + assign C) / 3

-- Theorem statement
theorem all_points_same_number
  (h : inscribedCircleProperty assign) :
  ∀ P Q : Point, assign P = assign Q :=
sorry

end NUMINAMATH_CALUDE_all_points_same_number_l2271_227160


namespace NUMINAMATH_CALUDE_find_r_l2271_227106

theorem find_r (a b c r : ℝ) 
  (h1 : a * (b - c) / (b * (c - a)) = r)
  (h2 : b * (c - a) / (c * (b - a)) = r)
  (h3 : r > 0) :
  r = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_find_r_l2271_227106


namespace NUMINAMATH_CALUDE_max_sum_of_logs_l2271_227148

-- Define the logarithm function (base 2)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem max_sum_of_logs (x y : ℝ) (h1 : x + y = 4) (h2 : x > 0) (h3 : y > 0) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 4 → lg a + lg b ≤ lg 4) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 4 ∧ lg a + lg b = lg 4) :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_logs_l2271_227148


namespace NUMINAMATH_CALUDE_second_planner_cheaper_at_34_l2271_227166

/-- Represents the pricing model of an event planner -/
structure PricingModel where
  flatFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total cost for a given number of people -/
def totalCost (model : PricingModel) (people : ℕ) : ℕ :=
  model.flatFee + model.perPersonFee * people

/-- The pricing model of the first planner -/
def planner1 : PricingModel := { flatFee := 150, perPersonFee := 18 }

/-- The pricing model of the second planner -/
def planner2 : PricingModel := { flatFee := 250, perPersonFee := 15 }

/-- Theorem stating that 34 is the least number of people for which the second planner is cheaper -/
theorem second_planner_cheaper_at_34 :
  (∀ n : ℕ, n < 34 → totalCost planner1 n ≤ totalCost planner2 n) ∧
  (totalCost planner2 34 < totalCost planner1 34) :=
by sorry

end NUMINAMATH_CALUDE_second_planner_cheaper_at_34_l2271_227166


namespace NUMINAMATH_CALUDE_binomial_12_10_l2271_227161

theorem binomial_12_10 : Nat.choose 12 10 = 66 := by sorry

end NUMINAMATH_CALUDE_binomial_12_10_l2271_227161


namespace NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_eq_five_halves_l2271_227108

theorem sqrt_a_div_sqrt_b_eq_five_halves (a b : ℝ) 
  (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (25*a)/(61*b)) : 
  Real.sqrt a / Real.sqrt b = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_eq_five_halves_l2271_227108


namespace NUMINAMATH_CALUDE_system_solutions_l2271_227171

/-- The polynomial f(t) = t³ - 4t² - 16t + 60 -/
def f (t : ℤ) : ℤ := t^3 - 4*t^2 - 16*t + 60

/-- The system of equations -/
def system (x y z : ℤ) : Prop :=
  f x = y ∧ f y = z ∧ f z = x

/-- The theorem stating the only integer solutions to the system -/
theorem system_solutions :
  ∀ x y z : ℤ, system x y z ↔ (x = 3 ∧ y = 3 ∧ z = 3) ∨ 
                               (x = 5 ∧ y = 5 ∧ z = 5) ∨ 
                               (x = -4 ∧ y = -4 ∧ z = -4) :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l2271_227171


namespace NUMINAMATH_CALUDE_total_discount_percentage_l2271_227113

-- Define the discounts
def initial_discount : ℝ := 0.3
def clearance_discount : ℝ := 0.2

-- Theorem statement
theorem total_discount_percentage : 
  (1 - (1 - initial_discount) * (1 - clearance_discount)) * 100 = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_discount_percentage_l2271_227113


namespace NUMINAMATH_CALUDE_experts_win_probability_value_l2271_227131

/-- The probability of Experts winning a single round -/
def p : ℝ := 0.6

/-- The probability of Audience winning a single round -/
def q : ℝ := 1 - p

/-- The current score of Experts -/
def experts_score : ℕ := 3

/-- The current score of Audience -/
def audience_score : ℕ := 4

/-- The number of wins needed to win the game -/
def wins_needed : ℕ := 6

/-- The probability that the Experts will eventually win the game -/
def experts_win_probability : ℝ := p^4 + 4 * p^3 * q

/-- Theorem stating that the probability of Experts winning is 0.4752 -/
theorem experts_win_probability_value : 
  experts_win_probability = 0.4752 := by sorry

end NUMINAMATH_CALUDE_experts_win_probability_value_l2271_227131


namespace NUMINAMATH_CALUDE_orangeade_ratio_l2271_227157

def orangeade_problem (orange_juice water_day1 : ℝ) : Prop :=
  let water_day2 := 2 * water_day1
  let price_day1 := 0.30
  let price_day2 := 0.20
  let volume_day1 := orange_juice + water_day1
  let volume_day2 := orange_juice + water_day2
  (volume_day1 * price_day1 = volume_day2 * price_day2) →
  (orange_juice = water_day1)

theorem orangeade_ratio :
  ∀ (orange_juice water_day1 : ℝ),
  orangeade_problem orange_juice water_day1 :=
sorry

end NUMINAMATH_CALUDE_orangeade_ratio_l2271_227157


namespace NUMINAMATH_CALUDE_triangle_theorem_l2271_227128

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : 2 * t.c * Real.cos t.B = 2 * t.a - Real.sqrt 3 * t.b) :
  t.C = π / 6 ∧ 
  (Real.cos t.B = 2 / 3 → Real.cos t.A = (Real.sqrt 5 - 2 * Real.sqrt 3) / 6) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2271_227128


namespace NUMINAMATH_CALUDE_angle_around_point_l2271_227194

/-- 
Given three angles around a point in a plane, where one angle is 130°, 
and one of the other angles (y) is 30° more than the third angle (x), 
prove that x = 100° and y = 130°.
-/
theorem angle_around_point (x y : ℝ) : 
  x + y + 130 = 360 →   -- Sum of angles around a point is 360°
  y = x + 30 →          -- y is 30° more than x
  x = 100 ∧ y = 130 :=  -- Conclusion: x = 100° and y = 130°
by sorry

end NUMINAMATH_CALUDE_angle_around_point_l2271_227194


namespace NUMINAMATH_CALUDE_circumradius_area_ratio_not_always_equal_l2271_227118

/-- Isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  perimeter : ℝ
  area : ℝ
  circumradius : ℝ

/-- Given two isosceles triangles with distinct sides, prove that the ratio of their circumradii
is not always equal to the ratio of their areas -/
theorem circumradius_area_ratio_not_always_equal
  (I II : IsoscelesTriangle)
  (h_distinct_base : I.base ≠ II.base)
  (h_distinct_side : I.side ≠ II.side) :
  ¬ ∀ (I II : IsoscelesTriangle),
    I.circumradius / II.circumradius = I.area / II.area :=
sorry

end NUMINAMATH_CALUDE_circumradius_area_ratio_not_always_equal_l2271_227118


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_A_subset_C_implies_a_greater_than_seven_l2271_227134

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Theorem for question 1
theorem union_A_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 10} := by sorry

-- Theorem for question 2
theorem complement_A_intersect_B : 
  (Set.univ \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10)} := by sorry

-- Theorem for question 3
theorem A_subset_C_implies_a_greater_than_seven (a : ℝ) : 
  A ⊆ C a → a > 7 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_A_subset_C_implies_a_greater_than_seven_l2271_227134


namespace NUMINAMATH_CALUDE_sequence_difference_theorem_l2271_227158

theorem sequence_difference_theorem (a : Fin 29 → ℤ) 
  (h_increasing : ∀ i j, i < j → a i < a j)
  (h_bound : ∀ k, k ≤ 22 → a (k + 7) - a k ≤ 13) :
  ∃ i j, a i - a j = 4 := by
sorry

end NUMINAMATH_CALUDE_sequence_difference_theorem_l2271_227158


namespace NUMINAMATH_CALUDE_problem_statement_l2271_227139

theorem problem_statement (a b : ℝ) (h : a - 2*b - 3 = 0) : 9 - 2*a + 4*b = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2271_227139


namespace NUMINAMATH_CALUDE_overlap_area_of_specific_triangles_l2271_227177

/-- A point in a 2D grid. -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A triangle defined by three points in a 2D grid. -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Calculate the area of a triangle given its base and height. -/
def triangleArea (base height : ℕ) : ℚ :=
  (base * height : ℚ) / 2

/-- The main theorem stating the area of overlap between two specific triangles. -/
theorem overlap_area_of_specific_triangles :
  let triangleA : GridTriangle := ⟨⟨0, 0⟩, ⟨2, 0⟩, ⟨2, 2⟩⟩
  let triangleB : GridTriangle := ⟨⟨0, 2⟩, ⟨2, 2⟩, ⟨0, 0⟩⟩
  triangleArea 2 2 = 2 := by sorry

end NUMINAMATH_CALUDE_overlap_area_of_specific_triangles_l2271_227177


namespace NUMINAMATH_CALUDE_determinant_problems_l2271_227144

def matrix1 : Matrix (Fin 3) (Fin 3) ℤ := !![3, 2, 1; 2, 5, 3; 3, 4, 3]

def matrix2 (a b c : ℤ) : Matrix (Fin 3) (Fin 3) ℤ := !![a, b, c; b, c, a; c, a, b]

theorem determinant_problems :
  (Matrix.det matrix1 = 8) ∧
  (∀ a b c : ℤ, Matrix.det (matrix2 a b c) = 3 * a * b * c - a^3 - b^3 - c^3) := by
  sorry

end NUMINAMATH_CALUDE_determinant_problems_l2271_227144


namespace NUMINAMATH_CALUDE_derivative_fifth_root_cube_l2271_227125

theorem derivative_fifth_root_cube (x : ℝ) (h : x ≠ 0) :
  deriv (λ x => x^(3/5)) x = 3 / (5 * x^(2/5)) :=
sorry

end NUMINAMATH_CALUDE_derivative_fifth_root_cube_l2271_227125


namespace NUMINAMATH_CALUDE_correct_factorization_l2271_227191

theorem correct_factorization (a : ℝ) : a^2 - 2*a + 1 = (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l2271_227191


namespace NUMINAMATH_CALUDE_largest_fraction_l2271_227141

theorem largest_fraction (a b c d e : ℚ) : 
  a = 2/5 → b = 3/6 → c = 5/10 → d = 7/15 → e = 8/20 →
  (b ≥ a ∧ b ≥ c ∧ b ≥ d ∧ b ≥ e) ∧
  (c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e) ∧
  b = c := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l2271_227141


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_probability_calculation_main_theorem_l2271_227154

/-- The probability of 2 zeros not being adjacent when 4 ones and 2 zeros are randomly arranged in a row -/
theorem zeros_not_adjacent_probability : ℚ :=
  2/3

/-- The total number of ways to arrange 4 ones and 2 zeros in a row -/
def total_arrangements : ℕ :=
  Nat.choose 6 2

/-- The number of arrangements where the 2 zeros are not adjacent -/
def non_adjacent_arrangements : ℕ :=
  Nat.choose 5 2

/-- The probability is the ratio of non-adjacent arrangements to total arrangements -/
theorem probability_calculation (h : zeros_not_adjacent_probability = non_adjacent_arrangements / total_arrangements) :
  zeros_not_adjacent_probability = 2/3 := by
  sorry

/-- The main theorem stating that the probability of 2 zeros not being adjacent is 2/3 -/
theorem main_theorem :
  zeros_not_adjacent_probability = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_probability_calculation_main_theorem_l2271_227154


namespace NUMINAMATH_CALUDE_second_train_speed_l2271_227172

/-- Calculates the speed of the second train given the parameters of two trains meeting. -/
theorem second_train_speed
  (length1 : ℝ) (length2 : ℝ) (speed1 : ℝ) (clear_time : ℝ)
  (h1 : length1 = 120) -- Length of first train in meters
  (h2 : length2 = 280) -- Length of second train in meters
  (h3 : speed1 = 42) -- Speed of first train in kmph
  (h4 : clear_time = 20 / 3600) -- Time to clear in hours
  : ∃ (speed2 : ℝ), speed2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_train_speed_l2271_227172


namespace NUMINAMATH_CALUDE_geometric_sequences_with_specific_differences_l2271_227175

/-- Two geometric sequences with the same first term and specific differences between their terms -/
theorem geometric_sequences_with_specific_differences :
  ∃ (a p q : ℚ),
    (a ≠ 0) ∧
    (p ≠ 0) ∧
    (q ≠ 0) ∧
    (a * p - a * q = 5) ∧
    (a * p^2 - a * q^2 = -5/4) ∧
    (a * p^3 - a * q^3 = 35/16) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequences_with_specific_differences_l2271_227175


namespace NUMINAMATH_CALUDE_opposites_equation_l2271_227105

theorem opposites_equation (x : ℝ) : (2 * x - 1 = -(-x + 5)) → (2 * x - 1 = x - 5) := by
  sorry

end NUMINAMATH_CALUDE_opposites_equation_l2271_227105


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_intersection_equals_B_iff_m_leq_1_l2271_227178

-- Define sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 3*m - 1}

-- Theorem for part (1)
theorem intersection_and_union_when_m_is_3 :
  (A ∩ B 3 = {x | -2 ≤ x ∧ x ≤ 2}) ∧
  (A ∪ B 3 = {x | -3 ≤ x ∧ x ≤ 8}) := by sorry

-- Theorem for part (2)
theorem intersection_equals_B_iff_m_leq_1 :
  ∀ m : ℝ, (A ∩ B m = B m) ↔ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_intersection_equals_B_iff_m_leq_1_l2271_227178


namespace NUMINAMATH_CALUDE_valentines_day_equality_l2271_227192

theorem valentines_day_equality (m d : ℕ) : 
  (∃ k : ℕ, 
    5 * m = 3 * k + 2 * (d - 3) ∧ 
    4 * d = 2 * k + 2 * (m - 2)) → 
  m = d :=
by
  sorry

end NUMINAMATH_CALUDE_valentines_day_equality_l2271_227192


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2271_227104

theorem complex_equation_sum (m n : ℝ) : 
  (m + n * Complex.I) * (4 - 2 * Complex.I) = 3 * Complex.I + 5 → m + n = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2271_227104


namespace NUMINAMATH_CALUDE_smallest_b_value_l2271_227133

theorem smallest_b_value (k a b : ℝ) (h1 : k > 1) (h2 : k < a) (h3 : a < b)
  (h4 : k + a ≤ b) (h5 : 1/a + 1/b ≤ 1/k) : b ≥ 2*k := by
  sorry

end NUMINAMATH_CALUDE_smallest_b_value_l2271_227133


namespace NUMINAMATH_CALUDE_complex_cube_sum_magnitude_l2271_227112

theorem complex_cube_sum_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs (z₁ + z₂) = 20) 
  (h2 : Complex.abs (z₁^2 + z₂^2) = 16) : 
  Complex.abs (z₁^3 + z₂^3) = 3520 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_sum_magnitude_l2271_227112


namespace NUMINAMATH_CALUDE_sequence_problem_l2271_227129

/-- Given a sequence {a_n} and an arithmetic sequence {b_n}, prove that a_6 = 33 -/
theorem sequence_problem (a b : ℕ → ℕ) : 
  a 1 = 3 →  -- First term of {a_n} is 3
  b 1 = 2 →  -- b_1 = 2
  b 3 = 6 →  -- b_3 = 6
  (∀ n : ℕ, n > 0 → b n = a (n + 1) - a n) →  -- b_n = a_{n+1} - a_n for n ∈ ℕ*
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, b (n + 1) = b n + d) →  -- {b_n} is an arithmetic sequence
  a 6 = 33 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l2271_227129


namespace NUMINAMATH_CALUDE_smallest_sum_of_four_consecutive_primes_divisible_by_five_l2271_227184

/-- A function that returns true if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if four consecutive numbers are all prime -/
def fourConsecutivePrimes (a b c d : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧ isPrime d ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1

/-- The main theorem -/
theorem smallest_sum_of_four_consecutive_primes_divisible_by_five :
  ∃ (a b c d : ℕ),
    fourConsecutivePrimes a b c d ∧
    (a + b + c + d) % 5 = 0 ∧
    a + b + c + d = 60 ∧
    ∀ (w x y z : ℕ),
      fourConsecutivePrimes w x y z →
      (w + x + y + z) % 5 = 0 →
      w + x + y + z ≥ 60 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_four_consecutive_primes_divisible_by_five_l2271_227184


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2271_227146

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = |x - 5| :=
by
  use 4
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2271_227146


namespace NUMINAMATH_CALUDE_product_divisible_by_49_l2271_227132

theorem product_divisible_by_49 (a b : ℕ) (h : 7 ∣ (a^2 + b^2)) : 49 ∣ (a * b) := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_49_l2271_227132


namespace NUMINAMATH_CALUDE_road_graveling_cost_l2271_227176

/-- Calculate the cost of graveling two intersecting roads on a rectangular lawn -/
theorem road_graveling_cost (lawn_length lawn_width road_width gravel_cost : ℝ) 
  (h1 : lawn_length = 90)
  (h2 : lawn_width = 60)
  (h3 : road_width = 10)
  (h4 : gravel_cost = 3) : 
  (lawn_length * road_width + lawn_width * road_width - road_width * road_width) * gravel_cost = 4200 := by
  sorry

#check road_graveling_cost

end NUMINAMATH_CALUDE_road_graveling_cost_l2271_227176


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2271_227122

/-- The function f(x) = x³ - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 0)

/-- Theorem: The equation of the tangent line to f(x) at (1, 0) is 2x - y - 2 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | 2 * x - y - 2 = 0} ↔
    y - f point.1 = f' point.1 * (x - point.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2271_227122


namespace NUMINAMATH_CALUDE_soup_problem_solution_l2271_227180

/-- Represents the number of people a can of soup can feed -/
structure CanCapacity where
  adults : Nat
  children : Nat

/-- Represents the problem setup -/
structure SoupProblem where
  capacity : CanCapacity
  totalCans : Nat
  childrenFed : Nat

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remainingAdults (problem : SoupProblem) : Nat :=
  let cansUsedForChildren := problem.childrenFed / problem.capacity.children
  let remainingCans := problem.totalCans - cansUsedForChildren
  remainingCans * problem.capacity.adults

/-- Proves that given the problem conditions, 16 adults can be fed with the remaining soup -/
theorem soup_problem_solution (problem : SoupProblem) 
  (h1 : problem.capacity = ⟨4, 6⟩) 
  (h2 : problem.totalCans = 8) 
  (h3 : problem.childrenFed = 24) : 
  remainingAdults problem = 16 := by
  sorry

#eval remainingAdults ⟨⟨4, 6⟩, 8, 24⟩

end NUMINAMATH_CALUDE_soup_problem_solution_l2271_227180


namespace NUMINAMATH_CALUDE_unique_half_value_l2271_227119

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x * y + f x) = x * f y + 2 * f x

/-- The theorem stating that f(1/2) has only one possible value, which is 1 -/
theorem unique_half_value (f : ℝ → ℝ) (hf : special_function f) : 
  ∃! v : ℝ, f (1/2) = v ∧ v = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_half_value_l2271_227119


namespace NUMINAMATH_CALUDE_janet_crayons_l2271_227173

/-- The number of crayons Michelle has initially -/
def michelle_initial : ℕ := 2

/-- The number of crayons Michelle has after Janet gives her all of her crayons -/
def michelle_final : ℕ := 4

/-- The number of crayons Janet has initially -/
def janet_initial : ℕ := michelle_final - michelle_initial

theorem janet_crayons : janet_initial = 2 := by sorry

end NUMINAMATH_CALUDE_janet_crayons_l2271_227173


namespace NUMINAMATH_CALUDE_largest_non_expressible_l2271_227101

/-- A function that checks if a number is composite -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- A function that checks if a number can be expressed as the sum of a multiple of 36 and a composite number -/
def IsExpressible (n : ℕ) : Prop :=
  ∃ (k : ℕ) (c : ℕ), k > 0 ∧ IsComposite c ∧ n = 36 * k + c

/-- Theorem stating that 304 is the largest number that cannot be expressed as the sum of a multiple of 36 and a composite number -/
theorem largest_non_expressible : 
  (¬ IsExpressible 304) ∧ (∀ n : ℕ, n > 304 → IsExpressible n) := by
  sorry

end NUMINAMATH_CALUDE_largest_non_expressible_l2271_227101


namespace NUMINAMATH_CALUDE_maxine_purchase_l2271_227181

theorem maxine_purchase (x y z : ℕ) : 
  x + y + z = 40 ∧ 
  50 * x + 400 * y + 500 * z = 10000 →
  x = 40 ∧ y = 0 ∧ z = 0 := by
sorry

end NUMINAMATH_CALUDE_maxine_purchase_l2271_227181


namespace NUMINAMATH_CALUDE_albert_has_two_snakes_l2271_227147

/-- Represents the number of snakes Albert has -/
def num_snakes : ℕ := 2

/-- Length of the garden snake in inches -/
def garden_snake_length : ℝ := 10.0

/-- Ratio of garden snake length to boa constrictor length -/
def snake_length_ratio : ℝ := 7.0

/-- Length of the boa constrictor in inches -/
def boa_constrictor_length : ℝ := 1.428571429

/-- Theorem stating that Albert has exactly 2 snakes given the conditions -/
theorem albert_has_two_snakes :
  num_snakes = 2 ∧
  garden_snake_length = 10.0 ∧
  boa_constrictor_length = garden_snake_length / snake_length_ratio ∧
  boa_constrictor_length = 1.428571429 :=
by sorry

end NUMINAMATH_CALUDE_albert_has_two_snakes_l2271_227147


namespace NUMINAMATH_CALUDE_difference_of_squares_l2271_227152

theorem difference_of_squares : 525^2 - 475^2 = 50000 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2271_227152


namespace NUMINAMATH_CALUDE_largest_negative_integer_l2271_227117

theorem largest_negative_integer : ∀ n : ℤ, n < 0 → n ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_largest_negative_integer_l2271_227117


namespace NUMINAMATH_CALUDE_not_enough_money_l2271_227155

/-- The cost of a single storybook in yuan -/
def storybook_cost : ℕ := 18

/-- The number of storybooks to be purchased -/
def num_books : ℕ := 12

/-- The available money in yuan -/
def available_money : ℕ := 200

/-- Theorem stating that the available money is not enough to buy the desired number of storybooks -/
theorem not_enough_money : storybook_cost * num_books > available_money := by
  sorry

end NUMINAMATH_CALUDE_not_enough_money_l2271_227155


namespace NUMINAMATH_CALUDE_expression_evaluation_l2271_227127

theorem expression_evaluation : (3^(2+3+4) - (3^2 * 3^3 + 3^4)) = 19359 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2271_227127


namespace NUMINAMATH_CALUDE_equation_solution_l2271_227126

theorem equation_solution : ∃ x : ℝ, 0.05 * x + 0.07 * (30 + x) = 14.7 ∧ x = 105 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2271_227126


namespace NUMINAMATH_CALUDE_max_value_of_f_l2271_227169

/-- The quadratic function f(x) = -3x^2 + 18x - 5 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 18 * x - 5

theorem max_value_of_f :
  ∃ (M : ℝ), M = 22 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2271_227169


namespace NUMINAMATH_CALUDE_blacksmith_horseshoe_solution_l2271_227189

/-- Represents the blacksmith's horseshoe problem --/
def horseshoe_problem (total_iron kg_per_horseshoe : ℕ)
  (num_farms horses_per_farm : ℕ)
  (num_stables horses_per_stable : ℕ)
  (horseshoes_per_horse : ℕ) : ℕ :=
  let total_horseshoes := total_iron / kg_per_horseshoe
  let farm_horses := num_farms * horses_per_farm
  let stable_horses := num_stables * horses_per_stable
  let total_order_horses := farm_horses + stable_horses
  let horseshoes_for_orders := total_order_horses * horseshoes_per_horse
  let remaining_horseshoes := total_horseshoes - horseshoes_for_orders
  remaining_horseshoes / horseshoes_per_horse

/-- Theorem stating the solution to the blacksmith's horseshoe problem --/
theorem blacksmith_horseshoe_solution :
  horseshoe_problem 400 2 2 2 2 5 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_blacksmith_horseshoe_solution_l2271_227189


namespace NUMINAMATH_CALUDE_margo_distance_l2271_227151

/-- Represents the total distance Margo traveled in miles -/
def total_distance : ℝ := 2.5

/-- Represents the time Margo took to walk to her friend's house in minutes -/
def walk_time : ℝ := 15

/-- Represents the time Margo took to jog back home in minutes -/
def jog_time : ℝ := 10

/-- Represents Margo's average speed for the entire trip in miles per hour -/
def average_speed : ℝ := 6

/-- Theorem stating that the total distance Margo traveled is 2.5 miles -/
theorem margo_distance :
  total_distance = average_speed * (walk_time + jog_time) / 60 :=
by sorry

end NUMINAMATH_CALUDE_margo_distance_l2271_227151


namespace NUMINAMATH_CALUDE_polynomial_value_at_three_l2271_227196

/-- Polynomial of degree 5 -/
def P (a b c d : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + d

theorem polynomial_value_at_three
  (a b c d : ℝ)
  (h1 : P a b c d 0 = -5)
  (h2 : P a b c d (-3) = 7) :
  P a b c d 3 = -17 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_three_l2271_227196


namespace NUMINAMATH_CALUDE_x_less_than_y_l2271_227107

theorem x_less_than_y (n : ℕ) (x y : ℝ) 
  (hn : n > 2) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxn : x^n = x + 1) 
  (hyn : y^(n+1) = y^3 + 1) : 
  x < y :=
by sorry

end NUMINAMATH_CALUDE_x_less_than_y_l2271_227107


namespace NUMINAMATH_CALUDE_fruits_in_good_condition_l2271_227195

theorem fruits_in_good_condition 
  (oranges : ℕ) 
  (bananas : ℕ) 
  (rotten_oranges_percent : ℚ) 
  (rotten_bananas_percent : ℚ) 
  (h1 : oranges = 600) 
  (h2 : bananas = 400) 
  (h3 : rotten_oranges_percent = 15/100) 
  (h4 : rotten_bananas_percent = 5/100) : 
  (oranges + bananas - (oranges * rotten_oranges_percent + bananas * rotten_bananas_percent)) / (oranges + bananas) = 89/100 := by
sorry

end NUMINAMATH_CALUDE_fruits_in_good_condition_l2271_227195


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l2271_227100

theorem sqrt_sum_fractions : 
  Real.sqrt ((1 : ℝ) / 25 + (1 : ℝ) / 36) = (Real.sqrt 61) / 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l2271_227100


namespace NUMINAMATH_CALUDE_remainder_2468135792_mod_101_l2271_227167

theorem remainder_2468135792_mod_101 : 2468135792 % 101 = 47 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2468135792_mod_101_l2271_227167


namespace NUMINAMATH_CALUDE_range_of_b_over_a_l2271_227188

theorem range_of_b_over_a (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : 5 - 3 * a ≤ b) (h2 : b ≤ 4 - a) (h3 : Real.log b ≥ a) :
  ∃ (x : ℝ), x = b / a ∧ e ≤ x ∧ x ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_range_of_b_over_a_l2271_227188


namespace NUMINAMATH_CALUDE_min_wire_length_for_specific_parallelepiped_l2271_227145

/-- The minimum length of wire needed to construct a rectangular parallelepiped -/
def wire_length (width length height : ℝ) : ℝ :=
  4 * (width + length + height)

/-- Theorem stating the minimum wire length for a specific rectangular parallelepiped -/
theorem min_wire_length_for_specific_parallelepiped :
  wire_length 10 8 5 = 92 := by
  sorry

end NUMINAMATH_CALUDE_min_wire_length_for_specific_parallelepiped_l2271_227145


namespace NUMINAMATH_CALUDE_dogs_barking_l2271_227124

theorem dogs_barking (initial_dogs : ℕ) (additional_dogs : ℕ) :
  initial_dogs = 30 →
  additional_dogs = 10 →
  initial_dogs + additional_dogs = 40 := by
sorry

end NUMINAMATH_CALUDE_dogs_barking_l2271_227124


namespace NUMINAMATH_CALUDE_items_washed_is_500_l2271_227165

/-- Calculates the total number of items washed given the number of loads, towels per load, and shirts per load. -/
def total_items_washed (loads : ℕ) (towels_per_load : ℕ) (shirts_per_load : ℕ) : ℕ :=
  loads * (towels_per_load + shirts_per_load)

/-- Proves that the total number of items washed is 500 given the specific conditions. -/
theorem items_washed_is_500 :
  total_items_washed 20 15 10 = 500 := by
  sorry

end NUMINAMATH_CALUDE_items_washed_is_500_l2271_227165


namespace NUMINAMATH_CALUDE_cone_lateral_area_l2271_227185

theorem cone_lateral_area (circumference : Real) (slant_height : Real) :
  circumference = 4 * Real.pi →
  slant_height = 3 →
  π * (circumference / (2 * π)) * slant_height = 6 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l2271_227185


namespace NUMINAMATH_CALUDE_f_uniquely_determined_l2271_227153

/-- A function from ℝ² to ℝ² defined as f(x, y) = (kx, y + b) -/
def f (k b : ℝ) : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (k * x, y + b)

/-- Theorem: If f(3, 1) = (6, 2), then k = 2 and b = 1 -/
theorem f_uniquely_determined (k b : ℝ) : 
  f k b (3, 1) = (6, 2) → k = 2 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_uniquely_determined_l2271_227153


namespace NUMINAMATH_CALUDE_strawberry_weight_theorem_l2271_227135

/-- The total weight of Marco's and his dad's strawberries -/
def total_weight (marco_weight : ℕ) (weight_difference : ℕ) : ℕ :=
  marco_weight + (marco_weight - weight_difference)

/-- Theorem: The total weight of strawberries is 47 pounds -/
theorem strawberry_weight_theorem (marco_weight : ℕ) (weight_difference : ℕ) 
  (h1 : marco_weight = 30)
  (h2 : weight_difference = 13) :
  total_weight marco_weight weight_difference = 47 := by
  sorry

#eval total_weight 30 13

end NUMINAMATH_CALUDE_strawberry_weight_theorem_l2271_227135


namespace NUMINAMATH_CALUDE_one_root_of_sum_equation_l2271_227143

/-- A reduced quadratic trinomial with two distinct roots -/
structure ReducedQuadraticTrinomial where
  b : ℝ
  c : ℝ
  has_distinct_roots : b^2 - 4*c > 0

/-- The discriminant of a quadratic trinomial -/
def discriminant (f : ReducedQuadraticTrinomial) : ℝ := f.b^2 - 4*f.c

/-- The quadratic function corresponding to a ReducedQuadraticTrinomial -/
def quad_function (f : ReducedQuadraticTrinomial) (x : ℝ) : ℝ := x^2 + f.b * x + f.c

/-- The theorem stating that f(x) + f(x - √D) = 0 has exactly one root -/
theorem one_root_of_sum_equation (f : ReducedQuadraticTrinomial) :
  ∃! x : ℝ, quad_function f x + quad_function f (x - Real.sqrt (discriminant f)) = 0 :=
sorry

end NUMINAMATH_CALUDE_one_root_of_sum_equation_l2271_227143


namespace NUMINAMATH_CALUDE_train_length_l2271_227114

/-- The length of a train that crosses a platform of equal length in one minute at 54 km/hr -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 54 → -- speed in km/hr
  time = 1 / 60 → -- time in hours (1 minute = 1/60 hour)
  length = speed * time / 2 → -- distance formula, divided by 2 due to equal lengths
  length = 450 / 1000 -- length in km (450m = 0.45km)
  := by sorry

end NUMINAMATH_CALUDE_train_length_l2271_227114


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2271_227182

theorem inequality_solution_set (m : ℝ) :
  let S := {x : ℝ | x^2 + (m - 1) * x - m > 0}
  (m = -1 → S = {x : ℝ | x ≠ 1}) ∧
  (m > -1 → S = {x : ℝ | x < -m ∨ x > 1}) ∧
  (m < -1 → S = {x : ℝ | x < 1 ∨ x > -m}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2271_227182


namespace NUMINAMATH_CALUDE_milena_age_l2271_227130

theorem milena_age :
  ∀ (milena_age grandmother_age grandfather_age : ℕ),
    grandmother_age = 9 * milena_age →
    grandfather_age = grandmother_age + 2 →
    grandfather_age - milena_age = 58 →
    milena_age = 7 := by
  sorry

end NUMINAMATH_CALUDE_milena_age_l2271_227130


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_71_l2271_227149

/-- Represents the product of a sequence following the pattern (n+1)/n from 5/3 to a/b --/
def sequence_product (a b : ℕ) : ℚ :=
  a / 3

theorem sum_of_a_and_b_is_71 (a b : ℕ) (h : sequence_product a b = 12) : a + b = 71 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_71_l2271_227149


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2271_227137

/-- The area of a square inscribed in a circle, which is itself inscribed in an equilateral triangle -/
theorem inscribed_square_area (s : ℝ) (h : s = 6) : 
  let r := s / (2 * Real.sqrt 3)
  let d := 2 * r
  let side := d / Real.sqrt 2
  side ^ 2 = 6 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2271_227137


namespace NUMINAMATH_CALUDE_steve_pages_written_l2271_227116

/-- Calculates the total number of pages Steve writes in a month -/
def total_pages_written (days_per_month : ℕ) (days_between_letters : ℕ) 
  (minutes_per_regular_letter : ℕ) (minutes_per_page : ℕ) 
  (minutes_for_long_letter : ℕ) : ℕ :=
  let regular_letters := days_per_month / days_between_letters
  let pages_per_regular_letter := minutes_per_regular_letter / minutes_per_page
  let regular_pages := regular_letters * pages_per_regular_letter
  let long_letter_pages := minutes_for_long_letter / (2 * minutes_per_page)
  regular_pages + long_letter_pages

theorem steve_pages_written :
  total_pages_written 30 3 20 10 80 = 24 := by sorry

end NUMINAMATH_CALUDE_steve_pages_written_l2271_227116


namespace NUMINAMATH_CALUDE_power_of_power_l2271_227179

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2271_227179


namespace NUMINAMATH_CALUDE_axis_of_symmetry_is_one_l2271_227159

/-- Given two perpendicular lines and a quadratic function, prove that the axis of symmetry is x=1 -/
theorem axis_of_symmetry_is_one 
  (a b : ℝ) 
  (h1 : ∀ x y : ℝ, b * x + a * y = 0 → x - 2 * y + 2 = 0 → (b * 1 + a * 0) * (1 * 1 + 2 * 0) = -1) 
  (f : ℝ → ℝ) 
  (h2 : ∀ x : ℝ, f x = a * x^2 - b * x + a) : 
  ∃ p : ℝ, p = 1 ∧ ∀ x : ℝ, f (p + x) = f (p - x) :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_is_one_l2271_227159


namespace NUMINAMATH_CALUDE_average_weight_decrease_l2271_227186

/-- Calculates the decrease in average weight when a new person is added to a group --/
theorem average_weight_decrease (initial_count : ℕ) (initial_average : ℝ) (new_weight : ℝ) :
  initial_count = 20 →
  initial_average = 60 →
  new_weight = 45 →
  let total_weight := initial_count * initial_average
  let new_total_weight := total_weight + new_weight
  let new_count := initial_count + 1
  let new_average := new_total_weight / new_count
  abs (initial_average - new_average - 0.71) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_decrease_l2271_227186


namespace NUMINAMATH_CALUDE_product_121_54_l2271_227138

theorem product_121_54 : 121 * 54 = 6534 := by
  sorry

end NUMINAMATH_CALUDE_product_121_54_l2271_227138


namespace NUMINAMATH_CALUDE_march_first_is_tuesday_l2271_227168

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a date in March -/
structure MarchDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Given that March 15 is a Tuesday, prove that March 1 is also a Tuesday -/
theorem march_first_is_tuesday (march15 : MarchDate) 
  (h15 : march15.day = 15 ∧ march15.dayOfWeek = DayOfWeek.Tuesday) :
  ∃ (march1 : MarchDate), march1.day = 1 ∧ march1.dayOfWeek = DayOfWeek.Tuesday :=
sorry

end NUMINAMATH_CALUDE_march_first_is_tuesday_l2271_227168


namespace NUMINAMATH_CALUDE_altitude_difference_example_l2271_227198

/-- The difference between the highest and lowest altitudes among three given altitudes -/
def altitude_difference (a b c : Int) : Int :=
  max a (max b c) - min a (min b c)

/-- Theorem stating that the altitude difference for the given values is 77 meters -/
theorem altitude_difference_example : altitude_difference (-102) (-80) (-25) = 77 := by
  sorry

end NUMINAMATH_CALUDE_altitude_difference_example_l2271_227198


namespace NUMINAMATH_CALUDE_blocks_needed_for_wall_l2271_227162

/-- Represents the dimensions of a wall -/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  length : Set ℚ
  height : ℕ

/-- Calculates the number of blocks needed for a wall with given conditions -/
def calculateBlocksNeeded (wall : WallDimensions) (block : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating that 540 blocks are needed for the given wall -/
theorem blocks_needed_for_wall :
  let wall := WallDimensions.mk 120 9
  let block := BlockDimensions.mk {2, 1.5, 1} 1
  calculateBlocksNeeded wall block = 540 := by
    sorry

end NUMINAMATH_CALUDE_blocks_needed_for_wall_l2271_227162


namespace NUMINAMATH_CALUDE_min_odd_counties_big_island_l2271_227142

/-- Represents a rectangular county with a diagonal road -/
structure County where
  has_diagonal_road : Bool

/-- Represents an island configuration -/
structure Island where
  counties : List County
  is_valid : Bool

/-- Checks if a given number of counties can form a valid Big Island configuration -/
def is_valid_big_island (n : Nat) : Prop :=
  ∃ (island : Island),
    island.counties.length = n ∧
    n % 2 = 1 ∧
    island.is_valid = true

/-- Theorem stating that 9 is the minimum odd number of counties for a valid Big Island -/
theorem min_odd_counties_big_island :
  (∀ k, k < 9 → k % 2 = 1 → ¬ is_valid_big_island k) ∧
  is_valid_big_island 9 := by
  sorry

end NUMINAMATH_CALUDE_min_odd_counties_big_island_l2271_227142
