import Mathlib

namespace NUMINAMATH_CALUDE_third_set_total_l2590_259034

/-- Represents a set of candies -/
structure CandySet where
  hard : ℕ
  chocolate : ℕ
  gummy : ℕ

/-- The problem setup -/
def candy_problem (set1 set2 set3 : CandySet) : Prop :=
  -- Total number of each type of candy is equal across all sets
  set1.hard + set2.hard + set3.hard = set1.chocolate + set2.chocolate + set3.chocolate ∧
  set1.hard + set2.hard + set3.hard = set1.gummy + set2.gummy + set3.gummy ∧
  -- First set conditions
  set1.chocolate = set1.gummy ∧
  set1.hard = set1.chocolate + 7 ∧
  -- Second set conditions
  set2.hard = set2.chocolate ∧
  set2.gummy = set2.hard - 15 ∧
  -- Third set condition
  set3.hard = 0

/-- The theorem to prove -/
theorem third_set_total (set1 set2 set3 : CandySet) 
  (h : candy_problem set1 set2 set3) : 
  set3.chocolate + set3.gummy = 29 := by
  sorry

end NUMINAMATH_CALUDE_third_set_total_l2590_259034


namespace NUMINAMATH_CALUDE_sqrt_calculation_l2590_259035

theorem sqrt_calculation : Real.sqrt 6 * Real.sqrt 3 + Real.sqrt 24 / Real.sqrt 6 - |(-3) * Real.sqrt 2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l2590_259035


namespace NUMINAMATH_CALUDE_smallest_four_digit_sum_20_l2590_259071

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is four digits -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- Theorem: 1999 is the smallest four-digit number whose digits sum to 20 -/
theorem smallest_four_digit_sum_20 : 
  (∀ n : ℕ, is_four_digit n → sum_of_digits n = 20 → 1999 ≤ n) ∧ 
  (is_four_digit 1999 ∧ sum_of_digits 1999 = 20) := by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_sum_20_l2590_259071


namespace NUMINAMATH_CALUDE_mom_tshirt_count_l2590_259043

/-- The number of t-shirts in a package -/
def shirts_per_package : ℕ := 6

/-- The number of packages Mom buys -/
def packages_bought : ℕ := 71

/-- Theorem: Mom will have 426 white t-shirts -/
theorem mom_tshirt_count : shirts_per_package * packages_bought = 426 := by
  sorry

end NUMINAMATH_CALUDE_mom_tshirt_count_l2590_259043


namespace NUMINAMATH_CALUDE_some_humans_are_pondering_l2590_259029

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Freshman : U → Prop)
variable (GradStudent : U → Prop)
variable (Human : U → Prop)
variable (Pondering : U → Prop)

-- State the theorem
theorem some_humans_are_pondering
  (h1 : ∀ x, Freshman x → Human x)
  (h2 : ∀ x, GradStudent x → Human x)
  (h3 : ∃ x, GradStudent x ∧ Pondering x) :
  ∃ x, Human x ∧ Pondering x :=
sorry

end NUMINAMATH_CALUDE_some_humans_are_pondering_l2590_259029


namespace NUMINAMATH_CALUDE_football_gear_cost_l2590_259075

theorem football_gear_cost (x : ℝ) 
  (h1 : x + x = 2 * x)  -- Shorts + T-shirt costs twice as much as shorts
  (h2 : x + 4 * x = 5 * x)  -- Shorts + boots costs five times as much as shorts
  (h3 : x + 2 * x = 3 * x)  -- Shorts + shin guards costs three times as much as shorts
  : x + x + 4 * x + 2 * x = 8 * x :=  -- Total cost is 8 times the cost of shorts
by sorry

end NUMINAMATH_CALUDE_football_gear_cost_l2590_259075


namespace NUMINAMATH_CALUDE_centroid_curve_area_centroid_curve_area_for_diameter_30_l2590_259064

/-- The area of the region bounded by the curve traced by the centroid of a triangle,
    where two vertices of the triangle are the endpoints of a circle's diameter,
    and the third vertex moves along the circle's circumference. -/
theorem centroid_curve_area (diameter : ℝ) : ℝ :=
  let radius := diameter / 2
  let centroid_radius := radius / 3
  let area := Real.pi * centroid_radius ^ 2
  ⌊area + 0.5⌋

/-- The area of the region bounded by the curve traced by the centroid of triangle ABC,
    where AB is a diameter of a circle with length 30 and C is a point on the circle,
    is approximately 79 (to the nearest positive integer). -/
theorem centroid_curve_area_for_diameter_30 :
  centroid_curve_area 30 = 79 := by
  sorry

end NUMINAMATH_CALUDE_centroid_curve_area_centroid_curve_area_for_diameter_30_l2590_259064


namespace NUMINAMATH_CALUDE_parametric_equations_of_Γ_polar_equation_of_perpendicular_line_l2590_259028

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the transformation
def transform (x y : ℝ) : ℝ × ℝ := (2*x, 3*y)

-- Define the curve Γ
def Γ (x y : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), unit_circle x₀ y₀ ∧ transform x₀ y₀ = (x, y)

-- Define the intersecting line l
def line_l (x y : ℝ) : Prop := 3*x + 2*y - 6 = 0

-- Theorem for parametric equations of Γ
theorem parametric_equations_of_Γ :
  ∀ t : ℝ, Γ (2 * Real.cos t) (3 * Real.sin t) := sorry

-- Theorem for polar equation of perpendicular line
theorem polar_equation_of_perpendicular_line :
  ∃ (P₁ P₂ : ℝ × ℝ),
    Γ P₁.1 P₁.2 ∧ Γ P₂.1 P₂.2 ∧
    line_l P₁.1 P₁.2 ∧ line_l P₂.1 P₂.2 ∧
    (∀ ρ θ : ℝ,
      4 * ρ * Real.cos θ - 6 * ρ * Real.sin θ + 5 = 0 ↔
      (∃ (k : ℝ),
        ρ * Real.cos θ = (P₁.1 + P₂.1) / 2 + k * 2 / 3 ∧
        ρ * Real.sin θ = (P₁.2 + P₂.2) / 2 - k * 3 / 2)) := sorry

end NUMINAMATH_CALUDE_parametric_equations_of_Γ_polar_equation_of_perpendicular_line_l2590_259028


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2590_259038

theorem min_value_of_expression :
  (∀ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023) ∧
  (∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 = 2023) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2590_259038


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l2590_259091

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x^2 - 4 * x - 6)

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max = 1 ∧
    min = 1 / Real.exp 8 :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l2590_259091


namespace NUMINAMATH_CALUDE_vectors_form_basis_l2590_259097

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, 7)

theorem vectors_form_basis : 
  LinearIndependent ℝ ![e₁, e₂] ∧ Submodule.span ℝ {e₁, e₂} = ⊤ :=
by sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l2590_259097


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l2590_259085

theorem parallelogram_side_length 
  (s : ℝ) 
  (h_positive : s > 0) 
  (h_angle : Real.cos (π / 3) = 1 / 2) 
  (h_area : (3 * s) * (s * Real.sin (π / 3)) = 27 * Real.sqrt 3) : 
  s = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l2590_259085


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2590_259012

/-- For a geometric sequence with first term 1, if the first term, the sum of first two terms,
    and 5 form an arithmetic sequence, then the common ratio is 2. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- a_n is a geometric sequence with common ratio q
  (a 1 = 1) →  -- First term is 1
  (S 2 = a 1 + a 2) →  -- S_2 is the sum of first two terms
  (S 2 - a 1 = 5 - S 2) →  -- a_1, S_2, and 5 form an arithmetic sequence
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2590_259012


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l2590_259047

/-- A circle inscribed in quadrilateral EFGH -/
structure InscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The circle is tangent to EF at R -/
  tangent_EF : True
  /-- The circle is tangent to GH at S -/
  tangent_GH : True
  /-- The circle is tangent to EH at T -/
  tangent_EH : True
  /-- ER = 25 -/
  ER : r = 25
  /-- RF = 35 -/
  RF : r = 35
  /-- GS = 40 -/
  GS : r = 40
  /-- SH = 20 -/
  SH : r = 20
  /-- ET = 45 -/
  ET : r = 45

/-- The square of the radius of the inscribed circle is 3600 -/
theorem inscribed_circle_radius_squared (c : InscribedCircle) : c.r^2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l2590_259047


namespace NUMINAMATH_CALUDE_cube_sum_of_roots_l2590_259073

theorem cube_sum_of_roots (p q r : ℝ) : 
  (p^3 - p^2 + p - 2 = 0) → 
  (q^3 - q^2 + q - 2 = 0) → 
  (r^3 - r^2 + r - 2 = 0) → 
  p^3 + q^3 + r^3 = 4 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_roots_l2590_259073


namespace NUMINAMATH_CALUDE_complex_number_modulus_l2590_259052

theorem complex_number_modulus (z : ℂ) : z = 1 / (Complex.I - 1) → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l2590_259052


namespace NUMINAMATH_CALUDE_cube_sum_greater_than_mixed_terms_l2590_259000

theorem cube_sum_greater_than_mixed_terms (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_greater_than_mixed_terms_l2590_259000


namespace NUMINAMATH_CALUDE_marathon_training_l2590_259086

theorem marathon_training (total_miles : ℝ) (day1_percent : ℝ) (day2_percent : ℝ) 
  (h1 : total_miles = 70)
  (h2 : day1_percent = 0.2)
  (h3 : day2_percent = 0.5) : 
  total_miles - (day1_percent * total_miles) - (day2_percent * (total_miles - day1_percent * total_miles)) = 28 := by
  sorry

end NUMINAMATH_CALUDE_marathon_training_l2590_259086


namespace NUMINAMATH_CALUDE_seating_theorem_l2590_259010

/-- The number of different seating arrangements for n people in m seats,
    where exactly two empty seats are adjacent -/
def seating_arrangements (m n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that for 6 seats and 3 people,
    the number of seating arrangements with exactly two adjacent empty seats is 72 -/
theorem seating_theorem : seating_arrangements 6 3 = 72 :=
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l2590_259010


namespace NUMINAMATH_CALUDE_fraction_and_decimal_representation_l2590_259084

theorem fraction_and_decimal_representation :
  (7 : ℚ) / 16 = 7 / 16 ∧ (100.45 : ℝ) = 100 + 4/10 + 5/100 :=
by sorry

end NUMINAMATH_CALUDE_fraction_and_decimal_representation_l2590_259084


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2590_259019

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 10 → 
  (a + b + c) / 3 = a + 20 → 
  (a + b + c) / 3 = c - 25 → 
  a + b + c = 45 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2590_259019


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2590_259046

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, x^2 > 1 → 1/x < 1) ∧
  (∃ x, 1/x < 1 ∧ x^2 ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2590_259046


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2590_259042

theorem min_value_sum_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_geometric_mean : Real.sqrt 3 = Real.sqrt (3^x * 3^(3*y))) :
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 1/(3*b) ≥ 1/x + 1/(3*y)) → 1/x + 1/(3*y) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2590_259042


namespace NUMINAMATH_CALUDE_three_means_sum_of_squares_l2590_259069

theorem three_means_sum_of_squares 
  (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 5)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 712.5 := by
sorry

end NUMINAMATH_CALUDE_three_means_sum_of_squares_l2590_259069


namespace NUMINAMATH_CALUDE_store_sales_problem_l2590_259014

theorem store_sales_problem (d : ℕ) : 
  (86 + 50 * d) / (d + 1) = 53 → d = 11 := by
  sorry

end NUMINAMATH_CALUDE_store_sales_problem_l2590_259014


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l2590_259009

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_root_existence 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hf1 : f a b c 1.4 = -0.24) 
  (hf2 : f a b c 1.5 = 0.25) :
  ∃ x₁ : ℝ, f a b c x₁ = 0 ∧ 1.4 < x₁ ∧ x₁ < 1.5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l2590_259009


namespace NUMINAMATH_CALUDE_if_A_then_all_short_answer_correct_l2590_259058

/-- Represents the condition for receiving an A grade -/
def receivedA (allShortAnswerCorrect : Bool) (multipleChoicePercentage : ℝ) : Prop :=
  allShortAnswerCorrect ∧ multipleChoicePercentage ≥ 90

/-- Proves that if a student received an A, they must have answered all short-answer questions correctly -/
theorem if_A_then_all_short_answer_correct 
  (student : String) 
  (studentReceivedA : Bool) 
  (studentAllShortAnswerCorrect : Bool) 
  (studentMultipleChoicePercentage : ℝ) : 
  (receivedA studentAllShortAnswerCorrect studentMultipleChoicePercentage → studentReceivedA) →
  (studentReceivedA → studentAllShortAnswerCorrect) :=
by sorry

end NUMINAMATH_CALUDE_if_A_then_all_short_answer_correct_l2590_259058


namespace NUMINAMATH_CALUDE_line_equation_l2590_259077

/-- Given a line passing through (b, 0) and (0, h), forming a triangle with area T' in the second quadrant where b > 0, prove that the equation of the line is -2T'x + b²y + 2T'b = 0. -/
theorem line_equation (b T' : ℝ) (h : ℝ) (hb : b > 0) : 
  ∃ (x y : ℝ → ℝ), ∀ t, -2 * T' * x t + b^2 * y t + 2 * T' * b = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l2590_259077


namespace NUMINAMATH_CALUDE_x_plus_four_value_l2590_259050

theorem x_plus_four_value (x t : ℝ) 
  (h1 : 6 * x + t = 4 * x - 9) 
  (h2 : t = 7) : 
  x + 4 = -4 := by
sorry

end NUMINAMATH_CALUDE_x_plus_four_value_l2590_259050


namespace NUMINAMATH_CALUDE_prime_square_sum_equation_l2590_259080

theorem prime_square_sum_equation (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ (x y z : ℕ), p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_square_sum_equation_l2590_259080


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l2590_259002

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (|x| - 1) / (x - 1) = 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l2590_259002


namespace NUMINAMATH_CALUDE_clock_hands_straight_in_day_l2590_259017

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Represents when the clock hands are straight -/
inductive ClockHandsStraight
  | coinciding
  | opposite

/-- Represents the position of the minute hand when the clock hands are straight -/
inductive MinuteHandPosition
  | zero_minutes
  | thirty_minutes

/-- The number of times the clock hands are straight in a day -/
def straight_hands_count : ℕ := 44

/-- Theorem stating that the clock hands are straight 44 times in a day -/
theorem clock_hands_straight_in_day :
  straight_hands_count = 44 :=
by sorry

end NUMINAMATH_CALUDE_clock_hands_straight_in_day_l2590_259017


namespace NUMINAMATH_CALUDE_new_average_production_l2590_259088

/-- Given the following conditions:
    1. The average daily production for the past n days was 50 units.
    2. Today's production is 90 units.
    3. The value of n is 19 days.
    Prove that the new average daily production is 52 units per day. -/
theorem new_average_production (n : ℕ) (prev_avg : ℝ) (today_prod : ℝ) :
  n = 19 ∧ prev_avg = 50 ∧ today_prod = 90 →
  (n * prev_avg + today_prod) / (n + 1) = 52 := by
  sorry

#check new_average_production

end NUMINAMATH_CALUDE_new_average_production_l2590_259088


namespace NUMINAMATH_CALUDE_inequality_region_l2590_259037

theorem inequality_region (x y : ℝ) : 
  ((x * y + 1) / (x + y))^2 < 1 ↔ 
  ((-1 < x ∧ x < 1 ∧ (y < -1 ∨ y > 1)) ∨ ((x < -1 ∨ x > 1) ∧ -1 < y ∧ y < 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_region_l2590_259037


namespace NUMINAMATH_CALUDE_sticks_per_pot_l2590_259032

/-- Given:
  * There are 466 pots
  * Each pot has 53 flowers
  * There are 109044 flowers and sticks in total
  Prove that there are 181 sticks in each pot -/
theorem sticks_per_pot (num_pots : ℕ) (flowers_per_pot : ℕ) (total_items : ℕ) :
  num_pots = 466 →
  flowers_per_pot = 53 →
  total_items = 109044 →
  (total_items - num_pots * flowers_per_pot) / num_pots = 181 := by
  sorry

#eval (109044 - 466 * 53) / 466  -- Should output 181

end NUMINAMATH_CALUDE_sticks_per_pot_l2590_259032


namespace NUMINAMATH_CALUDE_place_values_of_fours_l2590_259082

def number : ℕ := 40649003

theorem place_values_of_fours (n : ℕ) (h : n = number) :
  (n / 10000000 % 10 = 4 ∧ n / 10000000 * 10000000 = 40000000) ∧
  (n / 10000 % 10 = 4 ∧ n / 10000 % 10000 * 10000 = 40000) :=
sorry

end NUMINAMATH_CALUDE_place_values_of_fours_l2590_259082


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l2590_259031

/-- The number of ways to arrange four people from two teachers and four students,
    where the teachers must be selected and adjacent. -/
def arrangement_count : ℕ := 72

/-- The number of teachers -/
def teacher_count : ℕ := 2

/-- The number of students -/
def student_count : ℕ := 4

/-- The total number of people to be selected -/
def selection_count : ℕ := 4

theorem photo_arrangement_count :
  arrangement_count = 
    (teacher_count.factorial) *              -- Ways to arrange teachers
    (student_count.choose (selection_count - teacher_count)) * -- Ways to choose students
    ((selection_count - 1).factorial) :=     -- Ways to arrange teachers bundle and students
  by sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l2590_259031


namespace NUMINAMATH_CALUDE_remaining_milk_quantities_l2590_259023

/-- Represents the types of milk available in the store -/
inductive MilkType
  | Whole
  | LowFat
  | Almond

/-- Represents the initial quantities of milk bottles -/
def initial_quantity : MilkType → Nat
  | MilkType.Whole => 15
  | MilkType.LowFat => 12
  | MilkType.Almond => 8

/-- Represents Jason's purchase of whole milk -/
def jason_purchase : Nat := 5

/-- Represents Harry's purchase of low-fat milk -/
def harry_lowfat_purchase : Nat := 5  -- 4 bought + 1 free

/-- Represents Harry's purchase of almond milk -/
def harry_almond_purchase : Nat := 2

/-- Calculates the remaining quantity of a given milk type after purchases -/
def remaining_quantity (milk_type : MilkType) : Nat :=
  match milk_type with
  | MilkType.Whole => initial_quantity MilkType.Whole - jason_purchase
  | MilkType.LowFat => initial_quantity MilkType.LowFat - harry_lowfat_purchase
  | MilkType.Almond => initial_quantity MilkType.Almond - harry_almond_purchase

/-- Theorem stating the remaining quantities of milk bottles after purchases -/
theorem remaining_milk_quantities :
  remaining_quantity MilkType.Whole = 10 ∧
  remaining_quantity MilkType.LowFat = 7 ∧
  remaining_quantity MilkType.Almond = 6 := by
  sorry

end NUMINAMATH_CALUDE_remaining_milk_quantities_l2590_259023


namespace NUMINAMATH_CALUDE_initial_cards_eq_sum_l2590_259067

/-- The number of baseball cards Nell initially had -/
def initial_cards : ℕ := 242

/-- The number of cards Nell gave to Jeff -/
def cards_given : ℕ := 136

/-- The number of cards Nell has left -/
def cards_left : ℕ := 106

/-- Theorem stating that the initial number of cards is equal to the sum of cards given and cards left -/
theorem initial_cards_eq_sum : initial_cards = cards_given + cards_left := by
  sorry

end NUMINAMATH_CALUDE_initial_cards_eq_sum_l2590_259067


namespace NUMINAMATH_CALUDE_restore_exchange_rate_l2590_259078

/-- The exchange rate between Trade Federation's currency and Naboo's currency -/
structure ExchangeRate :=
  (rate : ℝ)

/-- The money supply of the Trade Federation -/
structure MoneySupply :=
  (supply : ℝ)

/-- The relationship between money supply changes and exchange rate changes -/
def money_supply_effect (ms_change : ℝ) : ℝ := 5 * ms_change

/-- The theorem stating the required change in money supply to restore the exchange rate -/
theorem restore_exchange_rate 
  (initial_rate : ExchangeRate)
  (new_rate : ExchangeRate)
  (money_supply : MoneySupply) :
  initial_rate.rate = 90 →
  new_rate.rate = 100 →
  (∀ (ms_change : ℝ), 
    ExchangeRate.rate (new_rate) * (1 - money_supply_effect ms_change / 100) = 
    ExchangeRate.rate (initial_rate)) →
  ∃ (ms_change : ℝ), ms_change = -2 :=
sorry

end NUMINAMATH_CALUDE_restore_exchange_rate_l2590_259078


namespace NUMINAMATH_CALUDE_power_difference_equals_negative_sixteen_million_l2590_259005

theorem power_difference_equals_negative_sixteen_million : (3^4)^3 - (4^3)^4 = -16245775 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equals_negative_sixteen_million_l2590_259005


namespace NUMINAMATH_CALUDE_percent_of_200_l2590_259004

theorem percent_of_200 : (25 / 100) * 200 = 50 := by sorry

end NUMINAMATH_CALUDE_percent_of_200_l2590_259004


namespace NUMINAMATH_CALUDE_ski_class_ratio_l2590_259056

theorem ski_class_ratio (b g : ℕ) : 
  b + g ≥ 66 →
  (b + 11 : ℤ) = (g - 13 : ℤ) →
  b ≠ 5 ∨ g ≠ 11 :=
by sorry

end NUMINAMATH_CALUDE_ski_class_ratio_l2590_259056


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l2590_259039

theorem smallest_constant_inequality (D : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2 ≥ D * (x - y)) ↔ D ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l2590_259039


namespace NUMINAMATH_CALUDE_keith_initial_cards_l2590_259059

/-- Represents the number of cards in Keith's collection --/
structure CardCollection where
  initial : ℕ
  added : ℕ
  remaining : ℕ

/-- Theorem stating the initial number of cards in Keith's collection --/
theorem keith_initial_cards (c : CardCollection) 
  (h1 : c.added = 8)
  (h2 : c.remaining = 46)
  (h3 : c.remaining * 2 = c.initial + c.added) :
  c.initial = 84 := by
  sorry

end NUMINAMATH_CALUDE_keith_initial_cards_l2590_259059


namespace NUMINAMATH_CALUDE_fraction_problem_l2590_259022

theorem fraction_problem (x : ℚ) : 
  x / (4 * x - 9) = 3 / 4 → x = 27 / 8 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l2590_259022


namespace NUMINAMATH_CALUDE_minimum_average_score_for_target_l2590_259092

def current_scores : List ℝ := [92, 81, 75, 65, 88]
def bonus_points : ℝ := 5
def target_increase : ℝ := 6

theorem minimum_average_score_for_target (new_test1 new_test2 : ℝ) :
  let current_avg := (current_scores.sum) / current_scores.length
  let new_avg := ((current_scores.sum + (new_test1 + bonus_points) + new_test2) / 
                  (current_scores.length + 2))
  let min_new_avg := (new_test1 + new_test2) / 2
  (new_avg = current_avg + target_increase) → min_new_avg ≥ 99 := by
  sorry

end NUMINAMATH_CALUDE_minimum_average_score_for_target_l2590_259092


namespace NUMINAMATH_CALUDE_difference_of_squares_l2590_259055

theorem difference_of_squares (x : ℝ) : x^2 - 25 = (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2590_259055


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l2590_259065

/-- Given a circle with area Q and circumference P, if Q/P = 10, then the radius is 20 -/
theorem circle_radius_from_area_circumference_ratio (Q P : ℝ) (hQ : Q > 0) (hP : P > 0) :
  Q / P = 10 → ∃ (r : ℝ), r > 0 ∧ Q = π * r^2 ∧ P = 2 * π * r ∧ r = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l2590_259065


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2590_259063

theorem unique_solution_condition (k : ℚ) : 
  (∃! x : ℚ, (x + 3) / (k * x - 2) = x) ↔ k = -3/4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2590_259063


namespace NUMINAMATH_CALUDE_quadratic_roots_opposite_signs_l2590_259026

theorem quadratic_roots_opposite_signs (p : ℝ) (hp : p > 2) :
  let f (x : ℝ) := 5 * x^2 - 4 * (p + 3) * x + 4 - p^2
  let x₁ := p + 2
  let x₂ := (-p + 2) / 5
  (f x₁ = 0) ∧ (f x₂ = 0) ∧ (x₁ * x₂ < 0) := by
  sorry

#check quadratic_roots_opposite_signs

end NUMINAMATH_CALUDE_quadratic_roots_opposite_signs_l2590_259026


namespace NUMINAMATH_CALUDE_interest_rate_calculation_interest_rate_value_l2590_259096

/-- The interest rate for Rs 100 over 8 years that produces the same interest as Rs 200 at 10% for 2 years -/
def interest_rate : ℝ := sorry

/-- The initial amount in rupees -/
def initial_amount : ℝ := 100

/-- The time period in years -/
def time_period : ℝ := 8

/-- The comparison amount in rupees -/
def comparison_amount : ℝ := 200

/-- The comparison interest rate -/
def comparison_rate : ℝ := 0.1

/-- The comparison time period in years -/
def comparison_time : ℝ := 2

theorem interest_rate_calculation : 
  initial_amount * interest_rate * time_period = 
  comparison_amount * comparison_rate * comparison_time :=
sorry

theorem interest_rate_value : interest_rate = 0.05 :=
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_interest_rate_value_l2590_259096


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2590_259021

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = 1 → z = (1 / 2 : ℂ) + Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2590_259021


namespace NUMINAMATH_CALUDE_min_value_of_f_l2590_259066

-- Define the function
def f (x : ℝ) : ℝ := x^4 - 4*x + 3

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem min_value_of_f : ∃ (x : ℝ), x ∈ interval ∧ f x = 0 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2590_259066


namespace NUMINAMATH_CALUDE_reconstruct_axes_and_unit_l2590_259020

-- Define the parabola
def parabola : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

-- Define the concept of constructible points
def constructible (p : ℝ × ℝ) : Prop := sorry

-- Define the concept of constructible lines
def constructibleLine (l : Set (ℝ × ℝ)) : Prop := sorry

-- Define the x-axis
def xAxis : Set (ℝ × ℝ) := {p | p.2 = 0}

-- Define the y-axis
def yAxis : Set (ℝ × ℝ) := {p | p.1 = 0}

-- Define the unit point (1, 1)
def unitPoint : ℝ × ℝ := (1, 1)

-- Theorem stating that the coordinate axes and unit length can be reconstructed
theorem reconstruct_axes_and_unit : 
  ∃ (x y : Set (ℝ × ℝ)) (u : ℝ × ℝ),
    constructibleLine x ∧ 
    constructibleLine y ∧ 
    constructible u ∧
    x = xAxis ∧ 
    y = yAxis ∧ 
    u = unitPoint :=
  sorry

end NUMINAMATH_CALUDE_reconstruct_axes_and_unit_l2590_259020


namespace NUMINAMATH_CALUDE_patio_rearrangement_l2590_259062

/-- Represents a rectangular patio layout --/
structure PatioLayout where
  rows : ℕ
  columns : ℕ
  total_tiles : ℕ

/-- Defines the conditions for a valid patio layout --/
def is_valid_layout (layout : PatioLayout) : Prop :=
  layout.total_tiles = layout.rows * layout.columns

/-- Defines the rearrangement of the patio --/
def rearranged_layout (original : PatioLayout) : PatioLayout :=
  { rows := original.total_tiles / (original.columns - 2)
  , columns := original.columns - 2
  , total_tiles := original.total_tiles }

/-- The main theorem to prove --/
theorem patio_rearrangement 
  (original : PatioLayout)
  (h_valid : is_valid_layout original)
  (h_rows : original.rows = 6)
  (h_total : original.total_tiles = 48) :
  (rearranged_layout original).rows - original.rows = 2 :=
sorry

end NUMINAMATH_CALUDE_patio_rearrangement_l2590_259062


namespace NUMINAMATH_CALUDE_hens_in_coop_l2590_259036

/-- Represents the chicken coop scenario --/
structure ChickenCoop where
  days : ℕ
  eggs_per_hen_per_day : ℕ
  boxes_filled : ℕ
  eggs_per_box : ℕ

/-- Calculates the number of hens in the chicken coop --/
def number_of_hens (coop : ChickenCoop) : ℕ :=
  (coop.boxes_filled * coop.eggs_per_box) / (coop.days * coop.eggs_per_hen_per_day)

/-- Theorem stating the number of hens in the specific scenario --/
theorem hens_in_coop : number_of_hens {
  days := 7,
  eggs_per_hen_per_day := 1,
  boxes_filled := 315,
  eggs_per_box := 6
} = 270 := by sorry

end NUMINAMATH_CALUDE_hens_in_coop_l2590_259036


namespace NUMINAMATH_CALUDE_theater_rows_l2590_259048

/-- Represents the number of rows in the theater. -/
def num_rows : ℕ := sorry

/-- Represents the number of students in the first condition. -/
def students_first_condition : ℕ := 30

/-- Represents the number of students in the second condition. -/
def students_second_condition : ℕ := 26

/-- Represents the minimum number of empty rows in the second condition. -/
def min_empty_rows : ℕ := 3

theorem theater_rows :
  (∀ (seating : Fin students_first_condition → Fin num_rows),
    ∃ (row : Fin num_rows) (s1 s2 : Fin students_first_condition),
      s1 ≠ s2 ∧ seating s1 = seating s2) ∧
  (∀ (seating : Fin students_second_condition → Fin num_rows),
    ∃ (empty_rows : Finset (Fin num_rows)),
      empty_rows.card ≥ min_empty_rows ∧
      ∀ (row : Fin num_rows),
        row ∈ empty_rows ↔ ∀ (s : Fin students_second_condition), seating s ≠ row) →
  num_rows = 29 :=
sorry

end NUMINAMATH_CALUDE_theater_rows_l2590_259048


namespace NUMINAMATH_CALUDE_total_rose_bushes_l2590_259089

theorem total_rose_bushes (rose_cost : ℕ) (aloe_cost : ℕ) (friend_roses : ℕ) (total_spent : ℕ) (aloe_count : ℕ) : 
  rose_cost = 75 → 
  friend_roses = 2 → 
  aloe_cost = 100 → 
  aloe_count = 2 → 
  total_spent = 500 → 
  (total_spent - aloe_count * aloe_cost) / rose_cost + friend_roses = 6 := by
sorry

end NUMINAMATH_CALUDE_total_rose_bushes_l2590_259089


namespace NUMINAMATH_CALUDE_point_C_coordinates_l2590_259060

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)

-- Define the line that point C is on
def line_C (x y : ℝ) : Prop := 3 * x - y + 3 = 0

-- Define the area of triangle ABC
def triangle_area (C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem point_C_coordinates :
  ∀ C : ℝ × ℝ,
  line_C C.1 C.2 →
  triangle_area C = 10 →
  C = (-1, 0) ∨ C = (5/3, 8) :=
sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l2590_259060


namespace NUMINAMATH_CALUDE_min_value_expression_l2590_259068

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a + 2/b) * (a + 2/b - 100) + (b + 2/a) * (b + 2/a - 100) ≥ -2500 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2590_259068


namespace NUMINAMATH_CALUDE_total_students_l2590_259027

/-- Given a row of students where a person is 4th from the left and 9th from the right,
    and there are 5 such rows with an equal number of students in each row,
    prove that the total number of students is 60. -/
theorem total_students (left_position : Nat) (right_position : Nat) (num_rows : Nat) 
  (h1 : left_position = 4)
  (h2 : right_position = 9)
  (h3 : num_rows = 5) :
  (left_position + right_position - 1) * num_rows = 60 := by
  sorry

#check total_students

end NUMINAMATH_CALUDE_total_students_l2590_259027


namespace NUMINAMATH_CALUDE_quadratic_rational_root_contradiction_l2590_259098

theorem quadratic_rational_root_contradiction (a b c : ℤ) (h_a_nonzero : a ≠ 0) 
  (h_rational_root : ∃ (p q : ℤ), q ≠ 0 ∧ a * (p / q)^2 + b * (p / q) + c = 0) 
  (h_all_odd : Odd a ∧ Odd b ∧ Odd c) : False :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_contradiction_l2590_259098


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2590_259018

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2590_259018


namespace NUMINAMATH_CALUDE_nine_workers_needed_workers_to_build_nine_cars_l2590_259044

/-- The number of workers needed to build a given number of cars in 9 days -/
def workers_needed (cars : ℕ) : ℕ :=
  cars

theorem nine_workers_needed : workers_needed 9 = 9 :=
by
  -- Proof goes here
  sorry

/-- Given condition: 7 workers can build 7 cars in 9 days -/
axiom seven_workers_seven_cars : workers_needed 7 = 7

-- The main theorem
theorem workers_to_build_nine_cars : ∃ w : ℕ, workers_needed 9 = w ∧ w = 9 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_nine_workers_needed_workers_to_build_nine_cars_l2590_259044


namespace NUMINAMATH_CALUDE_crayon_count_l2590_259054

theorem crayon_count (num_people : ℕ) (crayons_per_person : ℕ) (h1 : num_people = 3) (h2 : crayons_per_person = 8) : 
  num_people * crayons_per_person = 24 := by
  sorry

end NUMINAMATH_CALUDE_crayon_count_l2590_259054


namespace NUMINAMATH_CALUDE_smallest_c_value_l2590_259099

theorem smallest_c_value (a b c : ℤ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : b - a = c - b)  -- arithmetic progression
  (h4 : a * a = c * b)  -- geometric progression
  : c ≥ 4 ∧ ∃ (a' b' : ℤ), a' < b' ∧ b' < 4 ∧ b' - a' = 4 - b' ∧ a' * a' = 4 * b' := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l2590_259099


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l2590_259094

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 6 * a + 5 = 7) :
  ∃ (m : ℝ), m = -1 ∧ ∀ (x : ℝ), 8 * x^2 + 6 * x + 5 = 7 → 3 * x + 2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l2590_259094


namespace NUMINAMATH_CALUDE_probability_two_females_l2590_259079

def total_students : ℕ := 5
def female_students : ℕ := 3
def male_students : ℕ := 2
def students_to_select : ℕ := 2

theorem probability_two_females :
  (Nat.choose female_students students_to_select : ℚ) / 
  (Nat.choose total_students students_to_select : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_females_l2590_259079


namespace NUMINAMATH_CALUDE_ratio_equality_implies_fraction_value_l2590_259024

theorem ratio_equality_implies_fraction_value
  (a b c : ℝ)
  (h : a / 3 = b / 4 ∧ b / 4 = c / 5) :
  (a + b) / (b - c) = -7 :=
by sorry

end NUMINAMATH_CALUDE_ratio_equality_implies_fraction_value_l2590_259024


namespace NUMINAMATH_CALUDE_division_problem_l2590_259083

theorem division_problem (A : ℕ) : 
  (A / 6 = 3) ∧ (A % 6 = 2) → A = 20 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2590_259083


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_parallel_planes_l2590_259049

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- State the theorem
theorem lines_perpendicular_to_parallel_planes 
  (m n : Line) (α β : Plane) :
  m ≠ n →
  α ≠ β →
  parallel m n →
  perpendicular m α →
  perpendicular n β →
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_parallel_planes_l2590_259049


namespace NUMINAMATH_CALUDE_car_sale_profit_percentage_l2590_259072

theorem car_sale_profit_percentage (P : ℝ) : 
  let buying_price := 0.80 * P
  let selling_price := 1.16 * P
  ((selling_price - buying_price) / buying_price) * 100 = 45 := by
sorry

end NUMINAMATH_CALUDE_car_sale_profit_percentage_l2590_259072


namespace NUMINAMATH_CALUDE_expression_simplification_l2590_259081

theorem expression_simplification : 
  let x : ℝ := Real.sqrt 6 - Real.sqrt 2
  (x * (Real.sqrt 6 - x) + (x + Real.sqrt 5) * (x - Real.sqrt 5)) = 1 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2590_259081


namespace NUMINAMATH_CALUDE_subtraction_makes_equation_true_l2590_259045

theorem subtraction_makes_equation_true : 
  (5 - 2) + 6 - (4 - 3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_makes_equation_true_l2590_259045


namespace NUMINAMATH_CALUDE_exercise_book_count_l2590_259003

/-- Given a shop with pencils, pens, and exercise books in a specific ratio,
    calculate the number of exercise books based on the number of pencils. -/
theorem exercise_book_count (pencil_ratio : ℕ) (pen_ratio : ℕ) (book_ratio : ℕ) 
    (pencil_count : ℕ) (h1 : pencil_ratio = 10) (h2 : pen_ratio = 2) 
    (h3 : book_ratio = 3) (h4 : pencil_count = 120) : 
    (pencil_count / pencil_ratio) * book_ratio = 36 := by
  sorry

end NUMINAMATH_CALUDE_exercise_book_count_l2590_259003


namespace NUMINAMATH_CALUDE_total_pizza_slices_l2590_259074

theorem total_pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_pizzas = 2) (h2 : slices_per_pizza = 8) : 
  num_pizzas * slices_per_pizza = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_pizza_slices_l2590_259074


namespace NUMINAMATH_CALUDE_possible_k_values_l2590_259011

theorem possible_k_values (a b k : ℕ) (h1 : a > 0) (h2 : b > 0) :
  (b + 1 : ℚ) / a + (a + 1 : ℚ) / b = k →
  k = 3 ∨ k = 4 := by
sorry

end NUMINAMATH_CALUDE_possible_k_values_l2590_259011


namespace NUMINAMATH_CALUDE_paper_tearing_l2590_259008

theorem paper_tearing (n : ℕ) : 
  (∃ k : ℕ, 1 + 2 * k = 503) ∧ 
  (¬ ∃ k : ℕ, 1 + 2 * k = 2020) := by
  sorry

end NUMINAMATH_CALUDE_paper_tearing_l2590_259008


namespace NUMINAMATH_CALUDE_mango_purchase_l2590_259095

theorem mango_purchase (grapes_kg : ℕ) (grapes_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) :
  grapes_kg = 10 ∧ 
  grapes_rate = 70 ∧ 
  mango_rate = 55 ∧ 
  total_paid = 1195 →
  ∃ (mango_kg : ℕ), mango_kg = 9 ∧ grapes_kg * grapes_rate + mango_kg * mango_rate = total_paid :=
by sorry

end NUMINAMATH_CALUDE_mango_purchase_l2590_259095


namespace NUMINAMATH_CALUDE_pages_per_night_l2590_259013

/-- Given a book with 1200 pages read over 10.0 days, prove that 120 pages are read each night. -/
theorem pages_per_night (total_pages : ℕ) (reading_days : ℝ) :
  total_pages = 1200 → reading_days = 10.0 → (total_pages : ℝ) / reading_days = 120 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_night_l2590_259013


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l2590_259001

/-- The ratio of a cone's height to its radius when its volume is one-third of a sphere with the same radius -/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) : 
  (1 / 3) * ((4 / 3) * Real.pi * r^3) = (1 / 3) * Real.pi * r^2 * h → h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l2590_259001


namespace NUMINAMATH_CALUDE_coefficient_x3y5_proof_l2590_259007

/-- The coefficient of x^3y^5 in the expansion of (x+y)(x-y)^7 -/
def coefficient_x3y5 : ℤ := 14

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_x3y5_proof :
  coefficient_x3y5 = (binomial 7 4 : ℤ) - (binomial 7 5 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_proof_l2590_259007


namespace NUMINAMATH_CALUDE_sin_135_degrees_l2590_259041

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l2590_259041


namespace NUMINAMATH_CALUDE_girls_attending_sports_event_l2590_259016

theorem girls_attending_sports_event (total_students : ℕ) (attending_students : ℕ) 
  (h1 : total_students = 1500)
  (h2 : attending_students = 900)
  (h3 : ∃ (girls boys : ℕ), girls + boys = total_students ∧ 
                             (girls / 2 : ℚ) + (3 * boys / 5 : ℚ) = attending_students) :
  ∃ (girls : ℕ), girls / 2 = 500 := by
sorry

end NUMINAMATH_CALUDE_girls_attending_sports_event_l2590_259016


namespace NUMINAMATH_CALUDE_ripe_apples_weight_l2590_259070

/-- Given the total number of apples, the number of unripe apples, and the weight of each ripe apple,
    prove that the total weight of ripe apples is equal to the product of the number of ripe apples
    and the weight of each ripe apple. -/
theorem ripe_apples_weight
  (total_apples : ℕ)
  (unripe_apples : ℕ)
  (ripe_apple_weight : ℕ)
  (h1 : unripe_apples ≤ total_apples) :
  (total_apples - unripe_apples) * ripe_apple_weight =
    (total_apples - unripe_apples) * ripe_apple_weight :=
by sorry

end NUMINAMATH_CALUDE_ripe_apples_weight_l2590_259070


namespace NUMINAMATH_CALUDE_percentage_problem_l2590_259025

theorem percentage_problem : ∃ P : ℚ, P * 30 = 0.25 * 16 + 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2590_259025


namespace NUMINAMATH_CALUDE_inequality_proof_l2590_259051

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2*(a-1)*(b-1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2590_259051


namespace NUMINAMATH_CALUDE_intersection_point_d_l2590_259057

theorem intersection_point_d (d : ℝ) : 
  (∀ x y : ℝ, (y = x + d ∧ x = -y + d) → (x = d - 1 ∧ y = d)) → d = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_d_l2590_259057


namespace NUMINAMATH_CALUDE_article_selling_prices_l2590_259093

/-- Represents the selling price of an article -/
def selling_price (cost_price : ℚ) (profit_percentage : ℚ) : ℚ :=
  cost_price * (1 + profit_percentage / 100)

/-- Theorem stating the selling prices of three articles given their cost prices and profit/loss percentages -/
theorem article_selling_prices :
  let article1_sp := 500
  let article2_cp := 800
  let article3_cp := 1800
  (selling_price (article1_sp / 1.25) 25 = article1_sp) ∧
  (selling_price article2_cp (-25) = 600) ∧
  (selling_price article3_cp 50 = 2700) := by
  sorry


end NUMINAMATH_CALUDE_article_selling_prices_l2590_259093


namespace NUMINAMATH_CALUDE_root_line_discriminant_intersection_l2590_259053

/-- The discriminant curve in the pq-plane -/
def discriminant_curve (p q : ℝ) : Prop := 4 * p^3 + 27 * q^2 = 0

/-- The root line for a given value of a -/
def root_line (a p q : ℝ) : Prop := a * p + q + a^3 = 0

/-- The intersection points of the root line and the discriminant curve -/
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {(p, q) | discriminant_curve p q ∧ root_line a p q}

theorem root_line_discriminant_intersection (a : ℝ) :
  (a ≠ 0 → intersection_points a = {(-3 * a^2, 2 * a^3), (-3 * a^2 / 4, -a^3 / 4)}) ∧
  (a = 0 → intersection_points a = {(0, 0)}) := by
  sorry

end NUMINAMATH_CALUDE_root_line_discriminant_intersection_l2590_259053


namespace NUMINAMATH_CALUDE_ten_tables_seating_l2590_259030

/-- Calculates the number of people that can be seated at a given number of tables arranged in a row -/
def seatsInRow (numTables : ℕ) : ℕ :=
  if numTables = 0 then 0
  else if numTables = 1 then 6
  else if numTables = 2 then 10
  else if numTables = 3 then 14
  else 4 * numTables + 2

/-- Calculates the number of people that can be seated in a rectangular arrangement of tables -/
def seatsInRectangle (rows : ℕ) (tablesPerRow : ℕ) : ℕ :=
  rows * seatsInRow tablesPerRow

theorem ten_tables_seating :
  seatsInRectangle 2 5 = 80 :=
by sorry

end NUMINAMATH_CALUDE_ten_tables_seating_l2590_259030


namespace NUMINAMATH_CALUDE_total_liquid_consumed_l2590_259087

/-- The amount of cups in one pint -/
def cups_per_pint : ℝ := 2

/-- The amount of cups in one liter -/
def cups_per_liter : ℝ := 4.22675

/-- The amount of pints Elijah drank -/
def elijah_pints : ℝ := 8.5

/-- The amount of pints Emilio drank -/
def emilio_pints : ℝ := 9.5

/-- The amount of liters Isabella drank -/
def isabella_liters : ℝ := 3

/-- The total cups of liquid consumed by Elijah, Emilio, and Isabella -/
def total_cups : ℝ := elijah_pints * cups_per_pint + emilio_pints * cups_per_pint + isabella_liters * cups_per_liter

theorem total_liquid_consumed :
  total_cups = 48.68025 := by sorry

end NUMINAMATH_CALUDE_total_liquid_consumed_l2590_259087


namespace NUMINAMATH_CALUDE_morse_high_school_seniors_l2590_259033

/-- The number of seniors at Morse High School -/
def num_seniors : ℕ := 300

/-- The number of students in the lower grades (freshmen, sophomores, and juniors) -/
def num_lower_grades : ℕ := 900

/-- The percentage of seniors who have cars -/
def senior_car_percentage : ℚ := 1/2

/-- The percentage of lower grade students who have cars -/
def lower_grade_car_percentage : ℚ := 1/10

/-- The percentage of all students who have cars -/
def total_car_percentage : ℚ := 1/5

theorem morse_high_school_seniors :
  (num_seniors * senior_car_percentage + num_lower_grades * lower_grade_car_percentage : ℚ) = 
  ((num_seniors + num_lower_grades) * total_car_percentage : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_morse_high_school_seniors_l2590_259033


namespace NUMINAMATH_CALUDE_range_of_a_l2590_259076

def star_op (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) :
  (∀ x, star_op x (x - a) > 0 → -1 ≤ x ∧ x ≤ 1) →
  -2 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2590_259076


namespace NUMINAMATH_CALUDE_intersection_area_of_specific_circles_l2590_259090

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The area of intersection of two circles -/
def intersectionArea (c1 c2 : Circle) : ℝ := sorry

/-- The first circle centered at (3,0) with radius 3 -/
def circle1 : Circle := { center := (3, 0), radius := 3 }

/-- The second circle centered at (0,3) with radius 3 -/
def circle2 : Circle := { center := (0, 3), radius := 3 }

/-- Theorem stating the area of intersection of the two given circles -/
theorem intersection_area_of_specific_circles :
  intersectionArea circle1 circle2 = (9 * Real.pi - 18) / 2 := by sorry

end NUMINAMATH_CALUDE_intersection_area_of_specific_circles_l2590_259090


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l2590_259015

/-- The standard equation of a hyperbola sharing a focus with the parabola x² = 8y and having eccentricity 2 -/
theorem hyperbola_standard_equation :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →
  (∃ x₀ y₀ : ℝ, x₀^2 = 8*y₀ ∧ (x₀, y₀) = (0, 2)) →
  (a = 1 ∧ b^2 = 3) →
  ∀ x y : ℝ, y^2 - x^2 / 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l2590_259015


namespace NUMINAMATH_CALUDE_rest_area_milepost_l2590_259040

def first_exit : ℝ := 20
def seventh_exit : ℝ := 140

theorem rest_area_milepost : 
  let midpoint := (first_exit + seventh_exit) / 2
  midpoint = 80 := by sorry

end NUMINAMATH_CALUDE_rest_area_milepost_l2590_259040


namespace NUMINAMATH_CALUDE_grape_purchase_amount_l2590_259061

/-- The price of grapes per kg -/
def grape_price : ℕ := 70

/-- The price of mangoes per kg -/
def mango_price : ℕ := 55

/-- The number of kg of mangoes purchased -/
def mango_kg : ℕ := 9

/-- The total amount paid -/
def total_paid : ℕ := 1195

/-- The number of kg of grapes purchased -/
def grape_kg : ℕ := (total_paid - mango_price * mango_kg) / grape_price

theorem grape_purchase_amount : grape_kg = 10 := by
  sorry

end NUMINAMATH_CALUDE_grape_purchase_amount_l2590_259061


namespace NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l2590_259006

/-- The height of a right circular cylinder inscribed in a hemisphere -/
theorem cylinder_height_in_hemisphere (r_cylinder : ℝ) (r_hemisphere : ℝ) 
  (h_cylinder : r_cylinder = 3)
  (h_hemisphere : r_hemisphere = 7) :
  let h := Real.sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2)
  h = 2 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l2590_259006
