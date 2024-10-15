import Mathlib

namespace NUMINAMATH_CALUDE_weekly_production_total_l1264_126413

def john_rate : ℕ := 20
def jane_rate : ℕ := 15
def john_hours : List ℕ := [8, 6, 7, 5, 4]
def jane_hours : List ℕ := [7, 7, 6, 7, 8]

theorem weekly_production_total :
  (john_hours.map (· * john_rate)).sum + (jane_hours.map (· * jane_rate)).sum = 1125 := by
  sorry

end NUMINAMATH_CALUDE_weekly_production_total_l1264_126413


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l1264_126480

theorem ceiling_floor_difference : 
  ⌈(15 : ℝ) / 8 * (-34 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌊(-34 : ℝ) / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l1264_126480


namespace NUMINAMATH_CALUDE_optimal_investment_l1264_126474

/-- Represents an investment project with profit and loss rates -/
structure Project where
  maxProfitRate : Rat
  maxLossRate : Rat

/-- Represents an investment allocation -/
structure Investment where
  projectA : Rat
  projectB : Rat

def totalInvestment (i : Investment) : Rat :=
  i.projectA + i.projectB

def possibleLoss (p : Project) (i : Rat) : Rat :=
  i * p.maxLossRate

def possibleProfit (p : Project) (i : Rat) : Rat :=
  i * p.maxProfitRate

theorem optimal_investment
  (projectA : Project)
  (projectB : Project)
  (maxInvestment : Rat)
  (maxLoss : Rat)
  (h1 : projectA.maxProfitRate = 1)
  (h2 : projectB.maxProfitRate = 1/2)
  (h3 : projectA.maxLossRate = 3/10)
  (h4 : projectB.maxLossRate = 1/10)
  (h5 : maxInvestment = 100000)
  (h6 : maxLoss = 18000) :
  ∃ (i : Investment),
    totalInvestment i ≤ maxInvestment ∧
    possibleLoss projectA i.projectA + possibleLoss projectB i.projectB ≤ maxLoss ∧
    ∀ (j : Investment),
      totalInvestment j ≤ maxInvestment →
      possibleLoss projectA j.projectA + possibleLoss projectB j.projectB ≤ maxLoss →
      possibleProfit projectA i.projectA + possibleProfit projectB i.projectB ≥
      possibleProfit projectA j.projectA + possibleProfit projectB j.projectB ∧
    i.projectA = 40000 ∧
    i.projectB = 60000 :=
  sorry

#check optimal_investment

end NUMINAMATH_CALUDE_optimal_investment_l1264_126474


namespace NUMINAMATH_CALUDE_v_closed_under_multiplication_v_not_closed_under_add_cube_root_v_not_closed_under_division_v_not_closed_under_cube_cube_root_l1264_126455

-- Define the set v of cubes of positive integers
def v : Set ℕ := {n : ℕ | ∃ m : ℕ, n = m ^ 3}

-- Closure under multiplication
theorem v_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ v → b ∈ v → (a * b) ∈ v :=
sorry

-- Not closed under addition followed by cube root
theorem v_not_closed_under_add_cube_root :
  ∃ a b : ℕ, a ∈ v ∧ b ∈ v ∧ (∃ c : ℕ, c ^ 3 = a + b) → (∃ d : ℕ, d ^ 3 = a + b) :=
sorry

-- Not closed under division
theorem v_not_closed_under_division :
  ∃ a b : ℕ, a ∈ v ∧ b ∈ v ∧ b ≠ 0 → (a / b) ∉ v :=
sorry

-- Not closed under cubing followed by cube root
theorem v_not_closed_under_cube_cube_root :
  ∃ a : ℕ, a ∈ v ∧ (∃ b : ℕ, b ^ 3 = a ^ 3) → (∃ c : ℕ, c ^ 3 = a ^ 3) :=
sorry

end NUMINAMATH_CALUDE_v_closed_under_multiplication_v_not_closed_under_add_cube_root_v_not_closed_under_division_v_not_closed_under_cube_cube_root_l1264_126455


namespace NUMINAMATH_CALUDE_f_max_min_l1264_126436

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + x + 1

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3/2 }

theorem f_max_min :
  ∃ (max min : ℝ),
    (∀ x ∈ domain, f x ≤ max) ∧
    (∃ x ∈ domain, f x = max) ∧
    (∀ x ∈ domain, min ≤ f x) ∧
    (∃ x ∈ domain, f x = min) ∧
    max = 5/4 ∧
    min = 1/4 := by sorry

end NUMINAMATH_CALUDE_f_max_min_l1264_126436


namespace NUMINAMATH_CALUDE_hyperbola_center_l1264_126482

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) : 
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  f1 = (2, 3) → f2 = (-4, 7) → center = (-1, 5) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_center_l1264_126482


namespace NUMINAMATH_CALUDE_intersection_points_form_line_l1264_126492

theorem intersection_points_form_line (s : ℝ) :
  ∃ (x y : ℝ),
    (2 * x - 3 * y = 6 * s - 5) ∧
    (3 * x + y = 9 * s + 4) ∧
    (y = 3 * x + 16 / 11) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_form_line_l1264_126492


namespace NUMINAMATH_CALUDE_mary_fruit_expenses_l1264_126465

/-- The total cost of fruits Mary bought -/
def total_cost : ℚ := 34.72

/-- The cost of berries Mary bought -/
def berries_cost : ℚ := 11.08

/-- The cost of apples Mary bought -/
def apples_cost : ℚ := 14.33

/-- The cost of peaches Mary bought -/
def peaches_cost : ℚ := 9.31

/-- Theorem stating that the total cost is the sum of individual fruit costs -/
theorem mary_fruit_expenses : 
  total_cost = berries_cost + apples_cost + peaches_cost := by
  sorry

end NUMINAMATH_CALUDE_mary_fruit_expenses_l1264_126465


namespace NUMINAMATH_CALUDE_searchlight_dark_period_l1264_126466

/-- Given a searchlight that makes 3 revolutions per minute, 
    prove that if the probability of staying in the dark is 0.75, 
    then the duration of the dark period is 15 seconds. -/
theorem searchlight_dark_period 
  (revolutions_per_minute : ℝ) 
  (probability_dark : ℝ) 
  (h1 : revolutions_per_minute = 3) 
  (h2 : probability_dark = 0.75) : 
  (probability_dark * (60 / revolutions_per_minute)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_searchlight_dark_period_l1264_126466


namespace NUMINAMATH_CALUDE_sons_age_l1264_126429

/-- Given a father and son with specific age relationships, prove the son's age -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 25 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 23 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l1264_126429


namespace NUMINAMATH_CALUDE_f_order_l1264_126484

def f (x : ℝ) : ℝ := sorry

axiom f_even : ∀ x, f x = f (-x)
axiom f_periodic : ∀ x, f (x + 2) = f x
axiom f_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^(1/1998)

theorem f_order : f (101/17) < f (98/19) ∧ f (98/19) < f (104/15) := by sorry

end NUMINAMATH_CALUDE_f_order_l1264_126484


namespace NUMINAMATH_CALUDE_inequality_preservation_l1264_126437

theorem inequality_preservation (a b c : ℝ) (h : a > b) (h' : b > 0) :
  a + c > b + c := by
sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1264_126437


namespace NUMINAMATH_CALUDE_students_just_passed_l1264_126459

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) 
  (h_total : total = 300)
  (h_first : first_div_percent = 28 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_no_fail : first_div_percent + second_div_percent ≤ 1) :
  total - (total * first_div_percent).floor - (total * second_div_percent).floor = 54 :=
by sorry

end NUMINAMATH_CALUDE_students_just_passed_l1264_126459


namespace NUMINAMATH_CALUDE_vector_operation_l1264_126426

/-- Given two 2D vectors a and b, prove that the result of the vector operation is (-1, 2) -/
theorem vector_operation (a b : Fin 2 → ℝ) (ha : a = ![1, 1]) (hb : b = ![1, -1]) :
  (1/2 : ℝ) • a - (3/2 : ℝ) • b = ![(-1 : ℝ), 2] := by sorry

end NUMINAMATH_CALUDE_vector_operation_l1264_126426


namespace NUMINAMATH_CALUDE_collinear_vectors_l1264_126458

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![2, 3]

-- Define the sum vector
def sum_vector : Fin 2 → ℝ := ![3, 5]

-- Define the collinear vector
def collinear_vector : Fin 2 → ℝ := ![6, 10]

-- Theorem statement
theorem collinear_vectors :
  (∃ k : ℝ, ∀ i : Fin 2, collinear_vector i = k * sum_vector i) ∧
  (∀ i : Fin 2, sum_vector i = a i + b i) := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l1264_126458


namespace NUMINAMATH_CALUDE_torn_pages_theorem_l1264_126448

/-- Represents a set of consecutive pages torn from a book --/
structure TornPages where
  first : ℕ  -- First page number
  count : ℕ  -- Number of pages torn out

/-- The sum of consecutive integers from n to n + k - 1 --/
def sum_consecutive (n : ℕ) (k : ℕ) : ℕ :=
  k * (2 * n + k - 1) / 2

theorem torn_pages_theorem (pages : TornPages) :
  sum_consecutive pages.first pages.count = 344 →
  (344 = 2^3 * 43 ∧
   pages.first + (pages.first + pages.count - 1) = 43 ∧
   pages.count = 16) := by
  sorry


end NUMINAMATH_CALUDE_torn_pages_theorem_l1264_126448


namespace NUMINAMATH_CALUDE_sarah_bus_time_l1264_126486

-- Define the problem parameters
def leave_time : Nat := 7 * 60 + 45  -- 7:45 AM in minutes
def return_time : Nat := 17 * 60 + 15  -- 5:15 PM in minutes
def num_classes : Nat := 8
def class_duration : Nat := 45
def lunch_break : Nat := 30
def extracurricular_time : Nat := 90  -- 1 hour and 30 minutes in minutes

-- Define the theorem
theorem sarah_bus_time :
  let total_time := return_time - leave_time
  let school_time := num_classes * class_duration + lunch_break + extracurricular_time
  total_time - school_time = 90 := by
  sorry

end NUMINAMATH_CALUDE_sarah_bus_time_l1264_126486


namespace NUMINAMATH_CALUDE_baga_answer_variability_l1264_126408

/-- Represents a BAGA problem -/
structure BAGAProblem where
  conditions : Set String
  approach : String

/-- Represents the answer to a BAGA problem -/
structure BAGAAnswer where
  value : String

/-- Function that solves a BAGA problem -/
noncomputable def solveBagaProblem (problem : BAGAProblem) : BAGAAnswer :=
  sorry

/-- Theorem stating that small variations in BAGA problems can lead to different answers -/
theorem baga_answer_variability 
  (p1 p2 : BAGAProblem) 
  (h_small_diff : p1.conditions ≠ p2.conditions ∨ p1.approach ≠ p2.approach) : 
  ∃ (a1 a2 : BAGAAnswer), solveBagaProblem p1 = a1 ∧ solveBagaProblem p2 = a2 ∧ a1 ≠ a2 :=
sorry

end NUMINAMATH_CALUDE_baga_answer_variability_l1264_126408


namespace NUMINAMATH_CALUDE_even_polynomial_iff_product_with_negation_l1264_126433

/-- A polynomial over the complex numbers. -/
def ComplexPolynomial := ℂ → ℂ

/-- Predicate for even functions. -/
def IsEven (P : ComplexPolynomial) : Prop :=
  ∀ z : ℂ, P z = P (-z)

/-- The main theorem: A complex polynomial is even if and only if
    it can be expressed as the product of a polynomial and its negation. -/
theorem even_polynomial_iff_product_with_negation (P : ComplexPolynomial) :
  IsEven P ↔ ∃ Q : ComplexPolynomial, ∀ z : ℂ, P z = (Q z) * (Q (-z)) := by
  sorry

end NUMINAMATH_CALUDE_even_polynomial_iff_product_with_negation_l1264_126433


namespace NUMINAMATH_CALUDE_divided_square_area_l1264_126420

/-- A square divided into five rectangles of equal area -/
structure DividedSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The width of one rectangle -/
  rect_width : ℝ
  /-- The height of the central rectangle -/
  central_height : ℝ
  /-- All rectangles have equal area -/
  equal_area : ℝ
  /-- The given width of one rectangle is 5 -/
  width_condition : rect_width = 5
  /-- The square is divided into 5 rectangles -/
  division_condition : side = rect_width + 2 * central_height
  /-- Area of each rectangle -/
  area_condition : equal_area = rect_width * central_height
  /-- Total area of the square -/
  total_area : ℝ
  /-- Total area is the square of the side length -/
  area_calculation : total_area = side * side

/-- The theorem stating that the area of the divided square is 400 -/
theorem divided_square_area (s : DividedSquare) : s.total_area = 400 := by
  sorry

end NUMINAMATH_CALUDE_divided_square_area_l1264_126420


namespace NUMINAMATH_CALUDE_three_X_five_equals_two_l1264_126467

def X (a b : ℝ) : ℝ := b + 8 * a - a^3

theorem three_X_five_equals_two : X 3 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_three_X_five_equals_two_l1264_126467


namespace NUMINAMATH_CALUDE_laptop_to_phone_charger_ratio_l1264_126424

/-- Given a person with 4 phone chargers and 24 total chargers, 
    prove that the ratio of laptop chargers to phone chargers is 5. -/
theorem laptop_to_phone_charger_ratio : 
  ∀ (phone_chargers laptop_chargers : ℕ),
    phone_chargers = 4 →
    phone_chargers + laptop_chargers = 24 →
    laptop_chargers / phone_chargers = 5 := by
  sorry

end NUMINAMATH_CALUDE_laptop_to_phone_charger_ratio_l1264_126424


namespace NUMINAMATH_CALUDE_weekly_card_pack_size_l1264_126445

theorem weekly_card_pack_size (total_weeks : ℕ) (remaining_cards : ℕ) : 
  total_weeks = 52 →
  remaining_cards = 520 →
  (remaining_cards * 2) / total_weeks = 20 :=
by sorry

end NUMINAMATH_CALUDE_weekly_card_pack_size_l1264_126445


namespace NUMINAMATH_CALUDE_netflix_binge_watching_l1264_126421

theorem netflix_binge_watching (episode_length : ℕ) (daily_watch_time : ℕ) (days_to_finish : ℕ) : 
  episode_length = 20 →
  daily_watch_time = 120 →
  days_to_finish = 15 →
  (daily_watch_time * days_to_finish) / episode_length = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_netflix_binge_watching_l1264_126421


namespace NUMINAMATH_CALUDE_rain_probabilities_l1264_126491

/-- The probability of rain in place A -/
def prob_A : ℝ := 0.2

/-- The probability of rain in place B -/
def prob_B : ℝ := 0.3

/-- The probability of no rain in both places A and B -/
def prob_neither : ℝ := (1 - prob_A) * (1 - prob_B)

/-- The probability of rain in exactly one of places A or B -/
def prob_exactly_one : ℝ := prob_A * (1 - prob_B) + (1 - prob_A) * prob_B

/-- The probability of rain in at least one of places A or B -/
def prob_at_least_one : ℝ := 1 - prob_neither

/-- The probability of rain in at most one of places A or B -/
def prob_at_most_one : ℝ := prob_neither + prob_exactly_one

theorem rain_probabilities :
  prob_neither = 0.56 ∧
  prob_exactly_one = 0.38 ∧
  prob_at_least_one = 0.44 ∧
  prob_at_most_one = 0.94 := by
  sorry

end NUMINAMATH_CALUDE_rain_probabilities_l1264_126491


namespace NUMINAMATH_CALUDE_yankees_to_mets_ratio_l1264_126400

/-- Represents the number of fans for each baseball team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The total number of baseball fans in the town -/
def total_fans : ℕ := 390

/-- The theorem stating the ratio of NY Yankees fans to NY Mets fans -/
theorem yankees_to_mets_ratio (fc : FanCounts) : 
  fc.yankees = 156 ∧ fc.mets = 104 ∧ fc.red_sox = 130 →
  fc.yankees + fc.mets + fc.red_sox = total_fans →
  fc.mets * 5 = fc.red_sox * 4 →
  fc.yankees * 2 = fc.mets * 3 := by
  sorry

#check yankees_to_mets_ratio

end NUMINAMATH_CALUDE_yankees_to_mets_ratio_l1264_126400


namespace NUMINAMATH_CALUDE_nacho_triple_divya_age_l1264_126454

/-- Represents the number of years in the future when Nacho will be three times older than Divya -/
def future_years : ℕ := 10

/-- Divya's current age -/
def divya_age : ℕ := 5

/-- The sum of Nacho's and Divya's current ages -/
def total_current_age : ℕ := 40

/-- Nacho's current age -/
def nacho_age : ℕ := total_current_age - divya_age

theorem nacho_triple_divya_age : 
  nacho_age + future_years = 3 * (divya_age + future_years) :=
sorry

end NUMINAMATH_CALUDE_nacho_triple_divya_age_l1264_126454


namespace NUMINAMATH_CALUDE_circle_equation_l1264_126489

open Real

/-- A circle C in polar coordinates -/
structure PolarCircle where
  center : ℝ × ℝ
  passesThrough : ℝ × ℝ

/-- The equation of a line in polar form -/
def polarLine (θ₀ : ℝ) (k : ℝ) : ℝ → ℝ → Prop :=
  fun ρ θ ↦ ρ * sin (θ - θ₀) = k

theorem circle_equation (C : PolarCircle) 
  (h1 : C.passesThrough = (2 * sqrt 2, π/4))
  (h2 : C.center.1 = 2 ∧ C.center.2 = 0)
  (h3 : polarLine (π/3) (-sqrt 3) C.center.1 C.center.2) :
  ∀ θ, ∃ ρ, ρ = 4 * cos θ ∧ (ρ * cos θ - C.center.1)^2 + (ρ * sin θ - C.center.2)^2 = (2 * sqrt 2 - C.center.1)^2 + (2 * sqrt 2 - C.center.2)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1264_126489


namespace NUMINAMATH_CALUDE_range_of_trigonometric_function_l1264_126481

theorem range_of_trigonometric_function :
  ∀ x : ℝ, -1 ≤ Real.sin x * Real.cos x + Real.sin x + Real.cos x ∧ 
           Real.sin x * Real.cos x + Real.sin x + Real.cos x ≤ 1/2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_trigonometric_function_l1264_126481


namespace NUMINAMATH_CALUDE_julie_order_amount_l1264_126483

/-- The amount of food ordered by Julie -/
def julie_order : ℝ := 10

/-- The amount of food ordered by Letitia -/
def letitia_order : ℝ := 20

/-- The amount of food ordered by Anton -/
def anton_order : ℝ := 30

/-- The tip percentage -/
def tip_percentage : ℝ := 0.20

/-- The individual tip amount paid by each person -/
def individual_tip : ℝ := 4

theorem julie_order_amount :
  julie_order = 10 ∧
  letitia_order = 20 ∧
  anton_order = 30 ∧
  tip_percentage = 0.20 ∧
  individual_tip = 4 →
  tip_percentage * (julie_order + letitia_order + anton_order) = 3 * individual_tip :=
by sorry

end NUMINAMATH_CALUDE_julie_order_amount_l1264_126483


namespace NUMINAMATH_CALUDE_only_set_C_not_in_proportion_l1264_126456

def is_in_proportion (a b c d : ℝ) : Prop := a * d = b * c

theorem only_set_C_not_in_proportion :
  (is_in_proportion 4 8 5 10) ∧
  (is_in_proportion 2 (2 * Real.sqrt 5) (Real.sqrt 5) 5) ∧
  ¬(is_in_proportion 1 2 3 4) ∧
  (is_in_proportion 1 2 2 4) :=
by sorry

end NUMINAMATH_CALUDE_only_set_C_not_in_proportion_l1264_126456


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1264_126435

universe u

def U : Set ℕ := {1, 2, 3, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_theorem :
  (U \ M) ∪ N = {3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1264_126435


namespace NUMINAMATH_CALUDE_coal_shoveling_ratio_l1264_126469

/-- Represents the coal shoveling scenario -/
structure CoalScenario where
  people : ℕ
  days : ℕ
  coal : ℕ

/-- Calculates the daily rate of coal shoveling -/
def daily_rate (s : CoalScenario) : ℚ :=
  s.coal / (s.people * s.days)

theorem coal_shoveling_ratio :
  let original := CoalScenario.mk 10 10 10000
  let new := CoalScenario.mk (10 / 2) 80 40000
  daily_rate original = daily_rate new ∧
  (new.people : ℚ) / original.people = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_coal_shoveling_ratio_l1264_126469


namespace NUMINAMATH_CALUDE_circle_radius_from_tangents_l1264_126443

/-- Given a circle with diameter AB, tangents AD and BC, and a line through D and C
    intersecting the circle at E, prove that the radius is (c+d)/2 when AD = c and BC = d. -/
theorem circle_radius_from_tangents (c d : ℝ) (h : c ≠ d) :
  let circle : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ (x - 0)^2 + (y - 0)^2 = ((c + d)/2)^2}
  let A : ℝ × ℝ := (-(c + d)/2, 0)
  let B : ℝ × ℝ := ((c + d)/2, 0)
  let D : ℝ × ℝ := (-c, c)
  let C : ℝ × ℝ := (d, d)
  let E : ℝ × ℝ := (0, (c + d)/2)
  (∀ p ∈ circle, (p.1 - A.1)^2 + (p.2 - A.2)^2 = ((c + d)/2)^2) ∧
  (∀ p ∈ circle, (p.1 - B.1)^2 + (p.2 - B.2)^2 = ((c + d)/2)^2) ∧
  (D ∉ circle) ∧ (C ∉ circle) ∧
  ((D.1 - A.1) * (D.2 - A.2) + (D.1 - 0) * (D.2 - 0) = 0) ∧
  ((C.1 - B.1) * (C.2 - B.2) + (C.1 - 0) * (C.2 - 0) = 0) ∧
  (E ∈ circle) ∧
  (D.2 - A.2)/(D.1 - A.1) = (E.2 - D.2)/(E.1 - D.1) ∧
  (C.2 - B.2)/(C.1 - B.1) = (E.2 - C.2)/(E.1 - C.1) →
  (c + d)/2 = (c + d)/2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_from_tangents_l1264_126443


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l1264_126412

theorem rectangle_area_problem (total_area area1 area2 : ℝ) :
  total_area = 48 ∧ area1 = 24 ∧ area2 = 13 →
  total_area - (area1 + area2) = 11 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l1264_126412


namespace NUMINAMATH_CALUDE_parabola_solutions_l1264_126438

/-- A parabola defined by y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
def Parabola.y (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_solutions (p : Parabola) (m : ℝ) :
  p.y (-4) = m →
  p.y 0 = m →
  p.y 2 = 1 →
  p.y 4 = 0 →
  (∀ x : ℝ, p.y x = 0 ↔ x = 4 ∨ x = -8) :=
sorry

end NUMINAMATH_CALUDE_parabola_solutions_l1264_126438


namespace NUMINAMATH_CALUDE_quadratic_properties_l1264_126485

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

/-- Theorem stating that f satisfies the required conditions -/
theorem quadratic_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1264_126485


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l1264_126446

theorem min_value_sum_squares (x y z : ℝ) (h : x + 2*y + 3*z = 1) :
  ∃ (m : ℝ), (∀ a b c : ℝ, a + 2*b + 3*c = 1 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 = m) ∧
             (m = 1/14) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l1264_126446


namespace NUMINAMATH_CALUDE_opposite_of_negative_2022_l1264_126425

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_2022 : opposite (-2022) = 2022 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2022_l1264_126425


namespace NUMINAMATH_CALUDE_jello_bathtub_cost_is_270_l1264_126495

/-- Represents the cost in dollars to fill a bathtub with jello --/
def jelloBathtubCost (jelloMixPerPound : Real) (bathtubCapacity : Real) 
  (cubicFeetToGallons : Real) (poundsPerGallon : Real) (jelloMixCost : Real) : Real :=
  jelloMixPerPound * bathtubCapacity * cubicFeetToGallons * poundsPerGallon * jelloMixCost

/-- Theorem stating the cost to fill a bathtub with jello is $270 --/
theorem jello_bathtub_cost_is_270 :
  jelloBathtubCost 1.5 6 7.5 8 0.5 = 270 := by
  sorry

#eval jelloBathtubCost 1.5 6 7.5 8 0.5

end NUMINAMATH_CALUDE_jello_bathtub_cost_is_270_l1264_126495


namespace NUMINAMATH_CALUDE_ben_money_after_seven_days_l1264_126478

/-- Ben's daily allowance -/
def daily_allowance : ℕ := 50

/-- Ben's daily spending -/
def daily_spending : ℕ := 15

/-- Number of days -/
def num_days : ℕ := 7

/-- Ben's daily savings -/
def daily_savings : ℕ := daily_allowance - daily_spending

/-- Ben's total savings before mom's contribution -/
def initial_savings : ℕ := daily_savings * num_days

/-- Ben's savings after mom's contribution -/
def savings_after_mom : ℕ := 2 * initial_savings

/-- Dad's contribution -/
def dad_contribution : ℕ := 10

/-- Ben's final amount -/
def ben_final_amount : ℕ := savings_after_mom + dad_contribution

theorem ben_money_after_seven_days : ben_final_amount = 500 := by
  sorry

end NUMINAMATH_CALUDE_ben_money_after_seven_days_l1264_126478


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1264_126494

theorem fraction_to_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 208 / 10000 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1264_126494


namespace NUMINAMATH_CALUDE_ratio_equality_l1264_126471

theorem ratio_equality (a b : ℚ) (h : a / b = 7 / 6) : 6 * a = 7 * b := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1264_126471


namespace NUMINAMATH_CALUDE_percentage_less_than_500000_l1264_126487

-- Define the population categories
structure PopulationCategory where
  name : String
  percentage : ℝ

-- Define the theorem
theorem percentage_less_than_500000 (categories : List PopulationCategory)
  (h1 : categories.length = 3)
  (h2 : ∃ c ∈ categories, c.name = "less than 200,000" ∧ c.percentage = 35)
  (h3 : ∃ c ∈ categories, c.name = "200,000 to 499,999" ∧ c.percentage = 40)
  (h4 : ∃ c ∈ categories, c.name = "500,000 or more" ∧ c.percentage = 25)
  : (categories.filter (λ c => c.name = "less than 200,000" ∨ c.name = "200,000 to 499,999")).foldl (λ acc c => acc + c.percentage) 0 = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_less_than_500000_l1264_126487


namespace NUMINAMATH_CALUDE_f_domain_f_property_f_one_eq_zero_l1264_126447

/-- A function f with the given properties -/
def f : ℝ → ℝ :=
  sorry

theorem f_domain (x : ℝ) : x ≠ 0 → f x ≠ 0 :=
  sorry

theorem f_property (x₁ x₂ : ℝ) (h₁ : x₁ ≠ 0) (h₂ : x₂ ≠ 0) :
  f (x₁ * x₂) = f x₁ + f x₂ :=
  sorry

theorem f_one_eq_zero : f 1 = 0 :=
  sorry

end NUMINAMATH_CALUDE_f_domain_f_property_f_one_eq_zero_l1264_126447


namespace NUMINAMATH_CALUDE_hyperbola_equation_given_conditions_l1264_126419

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Check if a point is on a hyperbola -/
def point_on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  hyperbola_equation h p

theorem hyperbola_equation_given_conditions 
  (E : Hyperbola)
  (center : Point)
  (focus : Point)
  (N : Point)
  (h_center : center.x = 0 ∧ center.y = 0)
  (h_focus : focus.x = 3 ∧ focus.y = 0)
  (h_midpoint : N.x = -12 ∧ N.y = -15)
  (h_on_hyperbola : ∃ (A B : Point), 
    point_on_hyperbola E A ∧ 
    point_on_hyperbola E B ∧ 
    N.x = (A.x + B.x) / 2 ∧ 
    N.y = (A.y + B.y) / 2) :
  E.a^2 = 4 ∧ E.b^2 = 5 := by
  sorry

#check hyperbola_equation_given_conditions

end NUMINAMATH_CALUDE_hyperbola_equation_given_conditions_l1264_126419


namespace NUMINAMATH_CALUDE_initial_price_increase_l1264_126439

theorem initial_price_increase (P : ℝ) (x : ℝ) : 
  P * (1 + x / 100) * (1 - 10 / 100) = P * (1 + 12.5 / 100) → 
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_initial_price_increase_l1264_126439


namespace NUMINAMATH_CALUDE_mod_power_difference_l1264_126462

theorem mod_power_difference (n : ℕ) : 35^1723 - 16^1723 ≡ 1 [ZMOD 6] := by sorry

end NUMINAMATH_CALUDE_mod_power_difference_l1264_126462


namespace NUMINAMATH_CALUDE_line_inclination_angle_l1264_126468

theorem line_inclination_angle (x y : ℝ) :
  y - 3 = Real.sqrt 3 * (x - 4) →
  ∃ α : ℝ, 0 ≤ α ∧ α < π ∧ Real.tan α = Real.sqrt 3 ∧ α = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l1264_126468


namespace NUMINAMATH_CALUDE_certain_number_proof_l1264_126416

theorem certain_number_proof (x : ℝ) : (3 / 5) * x^2 = 126.15 → x = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1264_126416


namespace NUMINAMATH_CALUDE_luggage_per_passenger_l1264_126414

theorem luggage_per_passenger (total_passengers : ℕ) (total_bags : ℕ) 
  (h1 : total_passengers = 4) (h2 : total_bags = 32) : 
  total_bags / total_passengers = 8 := by
  sorry

end NUMINAMATH_CALUDE_luggage_per_passenger_l1264_126414


namespace NUMINAMATH_CALUDE_mother_daughter_age_relation_l1264_126461

theorem mother_daughter_age_relation :
  ∀ (mother_age daughter_age years_ago : ℕ),
  mother_age = 43 →
  daughter_age = 11 →
  mother_age - years_ago = 5 * (daughter_age - years_ago) →
  years_ago = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mother_daughter_age_relation_l1264_126461


namespace NUMINAMATH_CALUDE_mean_equality_problem_l1264_126497

theorem mean_equality_problem (x : ℝ) : 
  (8 + 16 + 24) / 3 = (10 + x) / 2 → x = 22 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_problem_l1264_126497


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l1264_126415

/-- Percentage of employees who attended the company picnic -/
def picnic_attendance (men_percentage : Real) (women_percentage : Real) 
  (men_attendance : Real) (women_attendance : Real) : Real :=
  men_percentage * men_attendance + (1 - men_percentage) * women_attendance

theorem company_picnic_attendance :
  picnic_attendance 0.45 0.55 0.20 0.40 = 0.31 := by
  sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l1264_126415


namespace NUMINAMATH_CALUDE_two_hour_charge_l1264_126441

/-- Represents the pricing structure for therapy sessions -/
structure TherapyPricing where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  hourDifference : firstHourCharge = additionalHourCharge + 25

/-- Calculates the total charge for a given number of therapy hours -/
def totalCharge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  if hours = 0 then 0
  else pricing.firstHourCharge + (hours - 1) * pricing.additionalHourCharge

/-- Theorem stating the total charge for 2 hours of therapy -/
theorem two_hour_charge (pricing : TherapyPricing) 
  (h : totalCharge pricing 5 = 250) : totalCharge pricing 2 = 115 := by
  sorry

end NUMINAMATH_CALUDE_two_hour_charge_l1264_126441


namespace NUMINAMATH_CALUDE_mass_percentage_H_in_NH4I_l1264_126410

-- Define atomic masses
def atomic_mass_N : ℝ := 14.01
def atomic_mass_H : ℝ := 1.01
def atomic_mass_I : ℝ := 126.90

-- Define the composition of NH4I
def NH4I_composition : Fin 3 → ℕ
  | 0 => 1  -- N
  | 1 => 4  -- H
  | 2 => 1  -- I
  | _ => 0

-- Define the molar mass of NH4I
def molar_mass_NH4I : ℝ :=
  NH4I_composition 0 * atomic_mass_N +
  NH4I_composition 1 * atomic_mass_H +
  NH4I_composition 2 * atomic_mass_I

-- Define the mass of hydrogen in NH4I
def mass_H_in_NH4I : ℝ := NH4I_composition 1 * atomic_mass_H

-- Theorem statement
theorem mass_percentage_H_in_NH4I :
  abs ((mass_H_in_NH4I / molar_mass_NH4I) * 100 - 2.79) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_mass_percentage_H_in_NH4I_l1264_126410


namespace NUMINAMATH_CALUDE_a_greater_than_b_l1264_126498

theorem a_greater_than_b : 
  let a := (-12) * (-23) * (-34) * (-45)
  let b := (-123) * (-234) * (-345)
  a > b := by
sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l1264_126498


namespace NUMINAMATH_CALUDE_cost_price_for_given_profit_l1264_126463

/-- Given a profit percentage, calculates the cost price as a percentage of the selling price -/
def cost_price_percentage (profit_percentage : Real) : Real :=
  100 - profit_percentage

/-- Theorem stating that when the profit percentage is 4.166666666666666%,
    the cost price is 95.83333333333334% of the selling price -/
theorem cost_price_for_given_profit :
  cost_price_percentage 4.166666666666666 = 95.83333333333334 := by
  sorry

#eval cost_price_percentage 4.166666666666666

end NUMINAMATH_CALUDE_cost_price_for_given_profit_l1264_126463


namespace NUMINAMATH_CALUDE_largest_perimeter_is_31_l1264_126417

/-- Represents a triangle with two fixed sides and one variable side --/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ

/-- Checks if the given lengths can form a valid triangle --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- Theorem stating the largest possible perimeter of the triangle --/
theorem largest_perimeter_is_31 :
  ∃ (t : Triangle), t.side1 = 7 ∧ t.side2 = 9 ∧ is_valid_triangle t ∧
  (∀ (t' : Triangle), t'.side1 = 7 ∧ t'.side2 = 9 ∧ is_valid_triangle t' →
    perimeter t' ≤ perimeter t) ∧
  perimeter t = 31 :=
sorry

end NUMINAMATH_CALUDE_largest_perimeter_is_31_l1264_126417


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1264_126473

theorem complex_equation_solution (i : ℂ) (z : ℂ) :
  i * i = -1 →
  (1 + i) * z = 1 + 3 * i →
  z = 2 + i := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1264_126473


namespace NUMINAMATH_CALUDE_sine_translation_l1264_126401

/-- Given a function g(x) obtained by translating y = sin(2x) to the right by π/12 units,
    prove that g(π/12) = 0 -/
theorem sine_translation (g : ℝ → ℝ) : 
  (∀ x, g x = Real.sin (2 * (x - π/12))) → g (π/12) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_translation_l1264_126401


namespace NUMINAMATH_CALUDE_complex_quadrant_l1264_126427

theorem complex_quadrant (z : ℂ) (h : (1 + 2*I)/z = 1 - I) : 
  z.re < 0 ∧ z.im > 0 := by
sorry

end NUMINAMATH_CALUDE_complex_quadrant_l1264_126427


namespace NUMINAMATH_CALUDE_tea_mixture_price_l1264_126440

/-- Given three varieties of tea with prices and mixing ratios, calculate the price of the mixture --/
theorem tea_mixture_price (p1 p2 p3 : ℚ) (r1 r2 r3 : ℚ) : 
  p1 = 126 → p2 = 135 → p3 = 173.5 → r1 = 1 → r2 = 1 → r3 = 2 →
  (p1 * r1 + p2 * r2 + p3 * r3) / (r1 + r2 + r3) = 152 := by
  sorry

#check tea_mixture_price

end NUMINAMATH_CALUDE_tea_mixture_price_l1264_126440


namespace NUMINAMATH_CALUDE_a_4_equals_28_l1264_126449

def S (n : ℕ) : ℕ := 4 * n^2

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem a_4_equals_28 : a 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_28_l1264_126449


namespace NUMINAMATH_CALUDE_line_inclination_angle_l1264_126452

def angle_of_inclination (x y : ℝ → ℝ) : ℝ := by sorry

theorem line_inclination_angle (t : ℝ) :
  let x := λ t : ℝ => 1 + Real.sqrt 3 * t
  let y := λ t : ℝ => 3 - 3 * t
  angle_of_inclination x y = 120 * π / 180 := by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l1264_126452


namespace NUMINAMATH_CALUDE_bishopArrangements_isPerfectSquare_l1264_126404

/-- The size of the chessboard -/
def boardSize : ℕ := 8

/-- The number of squares of one color on the board -/
def squaresPerColor : ℕ := boardSize * boardSize / 2

/-- The maximum number of non-threatening bishops on squares of one color -/
def maxBishopsPerColor : ℕ := boardSize

/-- The number of ways to arrange the maximum number of non-threatening bishops on an 8x8 chessboard -/
def totalArrangements : ℕ := (Nat.choose squaresPerColor maxBishopsPerColor) ^ 2

/-- Theorem stating that the number of arrangements is a perfect square -/
theorem bishopArrangements_isPerfectSquare : 
  ∃ n : ℕ, totalArrangements = n ^ 2 := by
sorry

end NUMINAMATH_CALUDE_bishopArrangements_isPerfectSquare_l1264_126404


namespace NUMINAMATH_CALUDE_product_equals_square_l1264_126457

theorem product_equals_square : 100 * 29.98 * 2.998 * 1000 = (2998 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l1264_126457


namespace NUMINAMATH_CALUDE_adoption_fee_is_correct_l1264_126442

/-- The adoption fee for an untrained seeing-eye dog. -/
def adoption_fee : ℝ := 150

/-- The weekly training cost for a seeing-eye dog. -/
def weekly_training_cost : ℝ := 250

/-- The number of weeks of training required. -/
def training_weeks : ℕ := 12

/-- The total cost of certification. -/
def certification_cost : ℝ := 3000

/-- The percentage of certification cost covered by insurance. -/
def insurance_coverage : ℝ := 0.9

/-- The total out-of-pocket cost for John. -/
def total_out_of_pocket : ℝ := 3450

/-- Theorem stating that the adoption fee is correct given the conditions. -/
theorem adoption_fee_is_correct : 
  adoption_fee + (weekly_training_cost * training_weeks) + 
  (certification_cost * (1 - insurance_coverage)) = total_out_of_pocket :=
by sorry

end NUMINAMATH_CALUDE_adoption_fee_is_correct_l1264_126442


namespace NUMINAMATH_CALUDE_santinos_mango_trees_l1264_126432

theorem santinos_mango_trees :
  let papaya_trees : ℕ := 2
  let papayas_per_tree : ℕ := 10
  let mangos_per_tree : ℕ := 20
  let total_fruits : ℕ := 80
  ∃ mango_trees : ℕ,
    papaya_trees * papayas_per_tree + mango_trees * mangos_per_tree = total_fruits ∧
    mango_trees = 3 :=
by sorry

end NUMINAMATH_CALUDE_santinos_mango_trees_l1264_126432


namespace NUMINAMATH_CALUDE_intersection_A_B_l1264_126406

def A : Set ℤ := {-3, -2, -1, 0, 1}
def B : Set ℤ := {x | x^2 - 4 = 0}

theorem intersection_A_B : A ∩ B = {-2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1264_126406


namespace NUMINAMATH_CALUDE_max_d_value_l1264_126423

def a (n : ℕ+) : ℕ := 101 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value : ∃ (k : ℕ+), d k = 3 ∧ ∀ (n : ℕ+), d n ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l1264_126423


namespace NUMINAMATH_CALUDE_cos_225_degrees_l1264_126428

theorem cos_225_degrees : Real.cos (225 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l1264_126428


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1264_126472

/-- Given a complex number z satisfying zi = (2+i)^2, prove that |z| = 5 -/
theorem complex_modulus_problem (z : ℂ) (h : z * Complex.I = (2 + Complex.I)^2) : 
  Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1264_126472


namespace NUMINAMATH_CALUDE_heart_ratio_two_four_four_two_l1264_126403

def heart (n m : ℕ) : ℕ := n^(3+m) * m^(2+n)

theorem heart_ratio_two_four_four_two :
  (heart 2 4 : ℚ) / (heart 4 2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_heart_ratio_two_four_four_two_l1264_126403


namespace NUMINAMATH_CALUDE_outer_boundary_diameter_is_44_l1264_126431

/-- The diameter of the circular fountain in feet. -/
def fountain_diameter : ℝ := 12

/-- The width of the garden ring in feet. -/
def garden_width : ℝ := 10

/-- The width of the walking path in feet. -/
def path_width : ℝ := 6

/-- The diameter of the circle forming the outer boundary of the walking path. -/
def outer_boundary_diameter : ℝ := fountain_diameter + 2 * (garden_width + path_width)

/-- Theorem stating that the diameter of the circle forming the outer boundary of the walking path is 44 feet. -/
theorem outer_boundary_diameter_is_44 : outer_boundary_diameter = 44 := by
  sorry

end NUMINAMATH_CALUDE_outer_boundary_diameter_is_44_l1264_126431


namespace NUMINAMATH_CALUDE_solve_for_y_l1264_126453

theorem solve_for_y (x y : ℚ) 
  (h1 : x = 103)
  (h2 : x^3 * y - 2 * x^2 * y + x * y - 100 * y = 1061500) : 
  y = 125 / 126 := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l1264_126453


namespace NUMINAMATH_CALUDE_median_list_i_equals_eight_l1264_126476

def list_i : List ℕ := [9, 2, 4, 7, 10, 11]
def list_ii : List ℕ := [3, 3, 4, 6, 7, 10]

def median (l : List ℕ) : ℚ := sorry
def mode (l : List ℕ) : ℕ := sorry

theorem median_list_i_equals_eight :
  median list_i = 8 :=
by
  have h1 : median list_i = median list_ii + mode list_ii := sorry
  sorry

#check median_list_i_equals_eight

end NUMINAMATH_CALUDE_median_list_i_equals_eight_l1264_126476


namespace NUMINAMATH_CALUDE_smallest_land_fraction_for_120_members_l1264_126496

/-- Represents a noble family with land inheritance rules -/
structure NobleFamily :=
  (total_members : ℕ)
  (has_original_plot : Bool)

/-- The smallest fraction of land a family member can receive -/
def smallest_land_fraction (family : NobleFamily) : ℚ :=
  1 / (2 * 3^39)

/-- Theorem stating the smallest possible land fraction for a family of 120 members -/
theorem smallest_land_fraction_for_120_members 
  (family : NobleFamily) 
  (h1 : family.total_members = 120) 
  (h2 : family.has_original_plot = true) : 
  smallest_land_fraction family = 1 / (2 * 3^39) := by
  sorry

end NUMINAMATH_CALUDE_smallest_land_fraction_for_120_members_l1264_126496


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1264_126407

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def A : Finset ℕ := {2, 3}
def B : Finset ℕ := {3, 5}

theorem intersection_with_complement : A ∩ (U \ B) = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1264_126407


namespace NUMINAMATH_CALUDE_original_number_proof_l1264_126422

theorem original_number_proof (x : ℝ) : 3 * ((2 * x)^2 + 5) = 129 → x = Real.sqrt 9.5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1264_126422


namespace NUMINAMATH_CALUDE_trig_identity_l1264_126464

theorem trig_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1264_126464


namespace NUMINAMATH_CALUDE_A_proper_superset_B_l1264_126450

def A : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def B : Set ℤ := {y | ∃ k : ℤ, y = 4 * k}

theorem A_proper_superset_B : A ⊃ B := by
  sorry

end NUMINAMATH_CALUDE_A_proper_superset_B_l1264_126450


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1264_126479

theorem fraction_to_decimal : (7 : ℚ) / 16 = (4375 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1264_126479


namespace NUMINAMATH_CALUDE_percentage_decrease_l1264_126402

theorem percentage_decrease (t : ℝ) (x : ℝ) : 
  t = 80 → 
  (t + 0.125 * t) - (t - x / 100 * t) = 30 → 
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_decrease_l1264_126402


namespace NUMINAMATH_CALUDE_day_after_2005_squared_days_l1264_126499

theorem day_after_2005_squared_days (start_day : ℕ) : 
  start_day % 7 = 0 → (start_day + 2005^2) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_day_after_2005_squared_days_l1264_126499


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l1264_126477

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define the axis of symmetry for a function
def axis_of_symmetry (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- State the theorem
theorem symmetry_of_shifted_even_function :
  is_even (λ x => f (x + 1)) → axis_of_symmetry f 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l1264_126477


namespace NUMINAMATH_CALUDE_no_rational_solution_l1264_126451

theorem no_rational_solution :
  ¬ ∃ (x y z t : ℚ) (n : ℕ), (x + y * Real.sqrt 2) ^ (2 * n) + (z + t * Real.sqrt 2) ^ (2 * n) = 5 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l1264_126451


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1260_l1264_126460

theorem sum_of_largest_and_smallest_prime_factors_of_1260 : ∃ (p q : Nat), 
  Nat.Prime p ∧ Nat.Prime q ∧ 
  p ∣ 1260 ∧ q ∣ 1260 ∧
  (∀ r : Nat, Nat.Prime r → r ∣ 1260 → p ≤ r ∧ r ≤ q) ∧
  p + q = 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1260_l1264_126460


namespace NUMINAMATH_CALUDE_asymptotic_function_part1_non_asymptotic_function_part2_l1264_126490

/-- Definition of asymptotic function -/
def is_asymptotic_function (f g : ℝ → ℝ) (p : ℝ) : Prop :=
  ∃ h : ℝ → ℝ, (∀ x ≥ 0, f x = g x + h x) ∧
  (Monotone (fun x ↦ -h x)) ∧
  (∀ x ≥ 0, 0 < h x ∧ h x ≤ p)

/-- Part I: Asymptotic function for f(x) = (x^2 + 2x + 3) / (x + 1) -/
theorem asymptotic_function_part1 :
  is_asymptotic_function (fun x ↦ (x^2 + 2*x + 3) / (x + 1)) (fun x ↦ x + 1) 2 :=
sorry

/-- Part II: Non-asymptotic function for f(x) = √(x^2 + 1) -/
theorem non_asymptotic_function_part2 (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ¬ is_asymptotic_function (fun x ↦ Real.sqrt (x^2 + 1)) (fun x ↦ a * x) p :=
sorry

end NUMINAMATH_CALUDE_asymptotic_function_part1_non_asymptotic_function_part2_l1264_126490


namespace NUMINAMATH_CALUDE_cookie_circle_radius_l1264_126418

theorem cookie_circle_radius (x y : ℝ) :
  x^2 + y^2 + 36 = 6*x + 12*y →
  ∃ (center : ℝ × ℝ), (x - center.1)^2 + (y - center.2)^2 = 3^2 := by
sorry

end NUMINAMATH_CALUDE_cookie_circle_radius_l1264_126418


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l1264_126430

/-- Represents a student in the line -/
inductive Student
  | boyA
  | boyB
  | girl1
  | girl2
  | girl3

/-- Represents a row of students -/
def Row := List Student

/-- Checks if exactly two of the three girls are adjacent in the row -/
def exactlyTwoGirlsAdjacent (row : Row) : Bool := sorry

/-- Checks if boy A is not at either end of the row -/
def boyANotAtEnds (row : Row) : Bool := sorry

/-- Generates all valid permutations of the students -/
def validPermutations : List Row := sorry

/-- Counts the number of valid arrangements -/
def countValidArrangements : Nat :=
  validPermutations.filter (λ row => exactlyTwoGirlsAdjacent row && boyANotAtEnds row) |>.length

theorem valid_arrangements_count :
  countValidArrangements = 36 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l1264_126430


namespace NUMINAMATH_CALUDE_unique_solution_for_a_l1264_126444

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2|

-- Define the function p(x, a)
def p (x a : ℝ) : ℝ := |x| + a

-- Define the domain of f
def D_f : Set ℝ := {x | x ≠ 2 ∧ x ≠ -1}

-- Theorem statement
theorem unique_solution_for_a (a : ℝ) :
  a ∈ Set.Ioo (-2) 0 ∪ Set.Ioo 0 2 ↔
  ∃! (x : ℝ), x ∈ D_f ∧ f x = p x a :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_a_l1264_126444


namespace NUMINAMATH_CALUDE_fraction_equals_870_l1264_126411

theorem fraction_equals_870 (a : ℕ+) :
  (a : ℚ) / ((a : ℚ) + 50) = 870 / 1000 → a = 335 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_870_l1264_126411


namespace NUMINAMATH_CALUDE_count_valid_numbers_eq_441_l1264_126493

/-- The count of valid digits for hundreds place (1-4, 7-9) -/
def valid_hundreds : Nat := 7

/-- The count of valid digits for tens place (0-4, 7-9) -/
def valid_tens : Nat := 7

/-- The count of valid digits for units place (1-9) -/
def valid_units : Nat := 9

/-- The count of three-digit whole numbers with no 5's and 6's in the tens and hundreds places -/
def count_valid_numbers : Nat := valid_hundreds * valid_tens * valid_units

theorem count_valid_numbers_eq_441 : count_valid_numbers = 441 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_numbers_eq_441_l1264_126493


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_coefficients_l1264_126409

theorem quadratic_roots_imply_coefficients (a b : ℝ) :
  (∀ x : ℝ, x^2 + (a + 1)*x + a*b = 0 ↔ x = -1 ∨ x = 4) →
  a = -4 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_coefficients_l1264_126409


namespace NUMINAMATH_CALUDE_problem_statements_l1264_126488

theorem problem_statements :
  (∃ x : ℝ, x^3 < 1) ∧
  ¬(∃ x : ℚ, x^2 = 2) ∧
  ¬(∀ x : ℕ, x^3 > x^2) ∧
  (∀ x : ℝ, x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l1264_126488


namespace NUMINAMATH_CALUDE_cricket_team_avg_age_l1264_126405

-- Define the team and its properties
structure CricketTeam where
  captain_age : ℝ
  wicket_keeper_age : ℝ
  num_bowlers : ℕ
  num_batsmen : ℕ
  team_avg_age : ℝ
  bowlers_avg_age : ℝ
  batsmen_avg_age : ℝ

-- Define the conditions
def team_conditions (team : CricketTeam) : Prop :=
  team.captain_age = 28 ∧
  team.wicket_keeper_age = team.captain_age + 3 ∧
  team.num_bowlers = 5 ∧
  team.num_batsmen = 4 ∧
  team.bowlers_avg_age = team.team_avg_age - 2 ∧
  team.batsmen_avg_age = team.team_avg_age + 3

-- Theorem statement
theorem cricket_team_avg_age (team : CricketTeam) :
  team_conditions team →
  team.team_avg_age = 30.5 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_avg_age_l1264_126405


namespace NUMINAMATH_CALUDE_soccer_team_non_players_l1264_126434

theorem soccer_team_non_players (total_players : ℕ) (starting_players : ℕ) (first_half_subs : ℕ) :
  total_players = 24 →
  starting_players = 11 →
  first_half_subs = 2 →
  total_players - (starting_players + first_half_subs + 2 * first_half_subs) = 7 :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_non_players_l1264_126434


namespace NUMINAMATH_CALUDE_pizza_slice_count_l1264_126470

/-- The total number of pizza slices given the conditions -/
def totalPizzaSlices (totalPizzas smallPizzaSlices largePizzaSlices : ℕ) : ℕ :=
  let smallPizzas := totalPizzas / 3
  let largePizzas := 2 * smallPizzas
  smallPizzas * smallPizzaSlices + largePizzas * largePizzaSlices

/-- Theorem stating that the total number of pizza slices is 384 -/
theorem pizza_slice_count :
  totalPizzaSlices 36 8 12 = 384 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_count_l1264_126470


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l1264_126475

theorem quadratic_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (x₁^2 - 5*x₁ + 11 = x₁ + 53) ∧ 
  (x₂^2 - 5*x₂ + 11 = x₂ + 53) ∧ 
  (x₁ ≠ x₂) ∧
  (|x₁ - x₂| = 2 * Real.sqrt 51) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l1264_126475
