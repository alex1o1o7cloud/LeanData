import Mathlib

namespace NUMINAMATH_CALUDE_no_real_roots_min_value_is_three_l3931_393178

-- Define the quadratic function
def f (m x : ℝ) : ℝ := x^2 - 2*m*x + m^2 + 3

-- Theorem 1: The quadratic equation has no real solutions for any m
theorem no_real_roots (m : ℝ) : ∀ x : ℝ, f m x ≠ 0 := by sorry

-- Theorem 2: The minimum value of the function is 3 for all m
theorem min_value_is_three (m : ℝ) : ∀ x : ℝ, f m x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_no_real_roots_min_value_is_three_l3931_393178


namespace NUMINAMATH_CALUDE_bears_per_shelf_l3931_393162

theorem bears_per_shelf (initial_stock : ℕ) (new_shipment : ℕ) (num_shelves : ℕ) 
  (h1 : initial_stock = 6)
  (h2 : new_shipment = 18)
  (h3 : num_shelves = 4) :
  (initial_stock + new_shipment) / num_shelves = 6 :=
by sorry

end NUMINAMATH_CALUDE_bears_per_shelf_l3931_393162


namespace NUMINAMATH_CALUDE_selling_price_optimal_l3931_393116

/-- Represents the selling price of toy A in yuan -/
def selling_price : ℝ := 65

/-- Represents the purchase price of toy A in yuan -/
def purchase_price : ℝ := 60

/-- Represents the maximum allowed profit margin -/
def max_profit_margin : ℝ := 0.4

/-- Represents the daily profit target in yuan -/
def profit_target : ℝ := 2500

/-- Calculates the number of units sold per day based on the selling price -/
def units_sold (x : ℝ) : ℝ := 1800 - 20 * x

/-- Calculates the profit per unit based on the selling price -/
def profit_per_unit (x : ℝ) : ℝ := x - purchase_price

/-- Calculates the total daily profit based on the selling price -/
def daily_profit (x : ℝ) : ℝ := profit_per_unit x * units_sold x

/-- Theorem stating that the selling price of 65 yuan results in the target profit
    while satisfying the profit margin constraint -/
theorem selling_price_optimal :
  daily_profit selling_price = profit_target ∧
  profit_per_unit selling_price / selling_price ≤ max_profit_margin :=
by sorry

end NUMINAMATH_CALUDE_selling_price_optimal_l3931_393116


namespace NUMINAMATH_CALUDE_sequence_problem_l3931_393183

/-- Given a sequence {a_n} and a geometric sequence {b_n}, prove a_10 = 64 -/
theorem sequence_problem (a b : ℕ → ℚ) : 
  a 1 = 1/8 →                           -- First term of sequence {a_n}
  b 5 = 2 →                             -- b_5 = 2 in geometric sequence {b_n}
  (∀ n, b n = a (n+1) / a n) →          -- Relation between a_n and b_n
  (∃ q, ∀ n, b n = 2 * q^(n-5)) →       -- b_n is a geometric sequence
  a 10 = 64 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l3931_393183


namespace NUMINAMATH_CALUDE_mean_study_days_is_4_05_l3931_393117

/-- Represents the study data for Ms. Rossi's class -/
structure StudyData where
  oneDay : Nat
  twoDays : Nat
  fourDays : Nat
  fiveDays : Nat
  sixDays : Nat

/-- Calculates the mean number of study days for the given data -/
def calculateMean (data : StudyData) : Float :=
  let totalDays := data.oneDay * 1 + data.twoDays * 2 + data.fourDays * 4 + data.fiveDays * 5 + data.sixDays * 6
  let totalStudents := data.oneDay + data.twoDays + data.fourDays + data.fiveDays + data.sixDays
  (totalDays.toFloat) / (totalStudents.toFloat)

/-- Theorem stating that the mean number of study days for Ms. Rossi's class is 4.05 -/
theorem mean_study_days_is_4_05 (data : StudyData) 
  (h1 : data.oneDay = 2)
  (h2 : data.twoDays = 4)
  (h3 : data.fourDays = 5)
  (h4 : data.fiveDays = 7)
  (h5 : data.sixDays = 4) :
  calculateMean data = 4.05 := by
  sorry

end NUMINAMATH_CALUDE_mean_study_days_is_4_05_l3931_393117


namespace NUMINAMATH_CALUDE_truncated_pyramid_volume_l3931_393118

/-- A regular truncated quadrilateral pyramid -/
structure TruncatedPyramid where
  upper_base : ℝ
  lower_base : ℝ

/-- The volume of a truncated pyramid -/
def volume (p : TruncatedPyramid) : ℝ := sorry

/-- A plane that divides the pyramid into two equal parts -/
structure DividingPlane where
  perpendicular_to_diagonal : Bool
  passes_through_upper_edge : Bool

theorem truncated_pyramid_volume
  (p : TruncatedPyramid)
  (d : DividingPlane)
  (h1 : p.upper_base = 1)
  (h2 : p.lower_base = 7)
  (h3 : d.perpendicular_to_diagonal = true)
  (h4 : d.passes_through_upper_edge = true)
  (h5 : ∃ (v : ℝ), volume { upper_base := p.upper_base, lower_base := v } = volume { upper_base := v, lower_base := p.lower_base }) :
  volume p = 38 / Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_truncated_pyramid_volume_l3931_393118


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3931_393181

theorem pure_imaginary_condition (k : ℝ) : 
  (2 * k^2 - 3 * k - 2 : ℂ) + (k^2 - 2 * k : ℂ) * Complex.I = Complex.I * (k^2 - 2 * k : ℂ) ↔ k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3931_393181


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l3931_393144

/-- The number of people at the table -/
def total_people : ℕ := 8

/-- The number of people Cara can choose from for her other neighbor -/
def available_neighbors : ℕ := total_people - 2

/-- The number of different possible pairs of people Cara could be sitting between -/
def seating_arrangements : ℕ := available_neighbors

theorem cara_seating_arrangements :
  seating_arrangements = 6 :=
sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l3931_393144


namespace NUMINAMATH_CALUDE_line_equation_l3931_393122

-- Define the point M
def M : ℝ × ℝ := (1, -2)

-- Define the line l
def l : Set (ℝ × ℝ) := sorry

-- Define points P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- State that P is on the x-axis
axiom P_on_x_axis : P.2 = 0

-- State that Q is on the y-axis
axiom Q_on_y_axis : Q.1 = 0

-- State that M is the midpoint of PQ
axiom M_is_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- State that P, Q, and M are on the line l
axiom P_on_l : P ∈ l
axiom Q_on_l : Q ∈ l
axiom M_on_l : M ∈ l

-- Theorem: The equation of line PQ is 2x - y - 4 = 0
theorem line_equation : ∀ (x y : ℝ), (x, y) ∈ l ↔ 2 * x - y - 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_equation_l3931_393122


namespace NUMINAMATH_CALUDE_smallest_consecutive_integer_sum_l3931_393179

/-- The sum of 15 consecutive positive integers that is a perfect square -/
def consecutiveIntegerSum (n : ℕ) : ℕ := 15 * (n + 7)

/-- The sum is a perfect square -/
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

theorem smallest_consecutive_integer_sum :
  (∃ n : ℕ, isPerfectSquare (consecutiveIntegerSum n)) →
  (∀ m : ℕ, isPerfectSquare (consecutiveIntegerSum m) → consecutiveIntegerSum m ≥ 225) :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_integer_sum_l3931_393179


namespace NUMINAMATH_CALUDE_sqrt_three_squared_plus_one_l3931_393113

theorem sqrt_three_squared_plus_one : (Real.sqrt 3)^2 + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_plus_one_l3931_393113


namespace NUMINAMATH_CALUDE_cubic_expression_value_l3931_393187

theorem cubic_expression_value : 
  let x : ℤ := -2
  (-2)^3 + (-2)^2 + 3*(-2) - 6 = -16 := by sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l3931_393187


namespace NUMINAMATH_CALUDE_equation_solution_l3931_393158

theorem equation_solution :
  let f (x : ℚ) := 1 - (3 + 2*x) / 4 = (x + 3) / 6
  ∃ (x : ℚ), f x ∧ x = -3/8 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3931_393158


namespace NUMINAMATH_CALUDE_total_rooms_in_hotel_l3931_393115

/-- Represents a wing of the hotel -/
structure Wing where
  floors : ℕ
  hallsPerFloor : ℕ
  singleRoomsPerHall : ℕ
  doubleRoomsPerHall : ℕ
  suitesPerHall : ℕ

/-- Calculates the total number of rooms in a wing -/
def totalRoomsInWing (w : Wing) : ℕ :=
  w.floors * w.hallsPerFloor * (w.singleRoomsPerHall + w.doubleRoomsPerHall + w.suitesPerHall)

/-- The first wing of the hotel -/
def wing1 : Wing :=
  { floors := 9
    hallsPerFloor := 6
    singleRoomsPerHall := 20
    doubleRoomsPerHall := 8
    suitesPerHall := 4 }

/-- The second wing of the hotel -/
def wing2 : Wing :=
  { floors := 7
    hallsPerFloor := 9
    singleRoomsPerHall := 25
    doubleRoomsPerHall := 10
    suitesPerHall := 5 }

/-- The third wing of the hotel -/
def wing3 : Wing :=
  { floors := 12
    hallsPerFloor := 4
    singleRoomsPerHall := 30
    doubleRoomsPerHall := 15
    suitesPerHall := 5 }

/-- Theorem stating the total number of rooms in the hotel -/
theorem total_rooms_in_hotel :
  totalRoomsInWing wing1 + totalRoomsInWing wing2 + totalRoomsInWing wing3 = 6648 := by
  sorry

end NUMINAMATH_CALUDE_total_rooms_in_hotel_l3931_393115


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l3931_393188

/-- A random variable following a normal distribution -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- Probability function for the normal distribution -/
noncomputable def probability (X : NormalDistribution) (a b : ℝ) : ℝ :=
  sorry

theorem normal_distribution_probability (X : NormalDistribution) 
  (h1 : X.μ = 4)
  (h2 : X.σ = 1)
  (h3 : probability X (X.μ - 2 * X.σ) (X.μ + 2 * X.σ) = 0.9544)
  (h4 : probability X (X.μ - X.σ) (X.μ + X.σ) = 0.6826) :
  probability X 5 6 = 0.1359 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l3931_393188


namespace NUMINAMATH_CALUDE_moon_radius_scientific_notation_l3931_393109

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) (hx : x > 0) : ScientificNotation :=
  sorry

theorem moon_radius_scientific_notation :
  toScientificNotation 1738000 (by norm_num) =
    ScientificNotation.mk 1.738 6 (by norm_num) (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_moon_radius_scientific_notation_l3931_393109


namespace NUMINAMATH_CALUDE_triangle_properties_l3931_393182

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating properties of a specific triangle -/
theorem triangle_properties (t : Triangle) (m : ℝ) : 
  (Real.sqrt 2 * Real.sin t.A = Real.sqrt (3 * Real.cos t.A)) →
  (t.a^2 - t.c^2 = t.b^2 - m * t.b * t.c) →
  (t.a = Real.sqrt 3) →
  (m = 1 ∧ 
   (∀ S : ℝ, S ≤ (3 * Real.sqrt 3) / 4 ∨ 
    ¬(∃ t' : Triangle, S = (t'.b * t'.c * Real.sin t'.A) / 2))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3931_393182


namespace NUMINAMATH_CALUDE_remainder_problem_l3931_393150

theorem remainder_problem (x : ℤ) : x % 82 = 5 → (x + 7) % 41 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3931_393150


namespace NUMINAMATH_CALUDE_inches_in_foot_l3931_393165

theorem inches_in_foot (room_side : ℝ) (room_area_sq_inches : ℝ) :
  room_side = 10 →
  room_area_sq_inches = 14400 →
  ∃ (inches_per_foot : ℝ), inches_per_foot = 12 ∧ 
    (room_side * inches_per_foot)^2 = room_area_sq_inches :=
by sorry

end NUMINAMATH_CALUDE_inches_in_foot_l3931_393165


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3931_393121

/-- Given a geometric sequence {a_n} with a₁ = 1 and common ratio q = 3,
    if the sum of the first t terms S_t = 364, then the t-th term a_t = 243 -/
theorem geometric_sequence_problem (t : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = 3 * a n) →  -- geometric sequence with q = 3
  a 1 = 1 →                    -- a₁ = 1
  (∀ n, S n = (a 1) * (1 - 3^n) / (1 - 3)) →  -- sum formula for geometric sequence
  S t = 364 →                  -- S_t = 364
  a t = 243 := by              -- a_t = 243
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3931_393121


namespace NUMINAMATH_CALUDE_fraction_calculation_and_comparison_l3931_393103

theorem fraction_calculation_and_comparison : 
  let x := (1/6 - 1/7) / (1/3 - 1/5)
  x = 5/28 ∧ 
  x ≠ 1/4 ∧ 
  x ≠ 1/3 ∧ 
  x ≠ 1/2 ∧ 
  x ≠ 2/5 ∧ 
  x ≠ 3/5 ∧
  x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_and_comparison_l3931_393103


namespace NUMINAMATH_CALUDE_sixth_employee_salary_l3931_393189

def employee_salaries : List ℝ := [1000, 2500, 3650, 1500, 2000]
def mean_salary : ℝ := 2291.67
def num_employees : ℕ := 6

theorem sixth_employee_salary :
  let total_salary := (mean_salary * num_employees)
  let known_salaries_sum := employee_salaries.sum
  total_salary - known_salaries_sum = 2100 := by
  sorry

end NUMINAMATH_CALUDE_sixth_employee_salary_l3931_393189


namespace NUMINAMATH_CALUDE_percentage_problem_l3931_393166

theorem percentage_problem (p : ℝ) : p * 50 = 0.15 → p = 0.003 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3931_393166


namespace NUMINAMATH_CALUDE_product_expansion_sum_l3931_393130

theorem product_expansion_sum (a b c d e : ℝ) :
  (∀ x : ℝ, (5*x^3 - 3*x^2 + x - 8)*(8 - 3*x) = a*x^4 + b*x^3 + c*x^2 + d*x + e) →
  16*a + 8*b + 4*c + 2*d + e = 44 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l3931_393130


namespace NUMINAMATH_CALUDE_celestia_badges_l3931_393111

def total_badges : ℕ := 83
def hermione_badges : ℕ := 14
def luna_badges : ℕ := 17

theorem celestia_badges : 
  total_badges - hermione_badges - luna_badges = 52 := by
  sorry

end NUMINAMATH_CALUDE_celestia_badges_l3931_393111


namespace NUMINAMATH_CALUDE_shells_added_l3931_393173

/-- Given that Jovana initially had 5 pounds of shells and now has 28 pounds,
    prove that she added 23 pounds of shells. -/
theorem shells_added (initial : ℕ) (final : ℕ) (h1 : initial = 5) (h2 : final = 28) :
  final - initial = 23 := by
  sorry

end NUMINAMATH_CALUDE_shells_added_l3931_393173


namespace NUMINAMATH_CALUDE_amare_fabric_needed_l3931_393124

/-- The amount of fabric Amare needs for the dresses -/
def fabric_needed (fabric_per_dress : ℝ) (num_dresses : ℕ) (fabric_owned : ℝ) : ℝ :=
  fabric_per_dress * num_dresses * 3 - fabric_owned

/-- Theorem stating the amount of fabric Amare needs -/
theorem amare_fabric_needed :
  fabric_needed 5.5 4 7 = 59 := by
  sorry

end NUMINAMATH_CALUDE_amare_fabric_needed_l3931_393124


namespace NUMINAMATH_CALUDE_total_dolls_l3931_393105

def doll_problem (vera_dolls : ℕ) (sophie_dolls : ℕ) (aida_dolls : ℕ) : Prop :=
  vera_dolls = 20 ∧
  sophie_dolls = 2 * vera_dolls ∧
  aida_dolls = 2 * sophie_dolls ∧
  aida_dolls + sophie_dolls + vera_dolls = 140

theorem total_dolls :
  ∃ (vera_dolls sophie_dolls aida_dolls : ℕ),
    doll_problem vera_dolls sophie_dolls aida_dolls :=
by
  sorry

end NUMINAMATH_CALUDE_total_dolls_l3931_393105


namespace NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l3931_393169

theorem tens_digit_of_2023_pow_2024_minus_2025 :
  (2023^2024 - 2025) % 100 / 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l3931_393169


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l3931_393123

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_11th_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + a 3 + a 5 + a 7 + a 9 = 10)
  (h_diff : a 8 ^ 2 - a 2 ^ 2 = 36) :
  a 11 = 11 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l3931_393123


namespace NUMINAMATH_CALUDE_student_arrangement_count_l3931_393191

def num_male_students : ℕ := 3
def num_female_students : ℕ := 3
def total_students : ℕ := num_male_students + num_female_students

def adjacent_female_students : ℕ := 2

def num_arrangements : ℕ := 432

theorem student_arrangement_count :
  (num_male_students = 3) →
  (num_female_students = 3) →
  (total_students = num_male_students + num_female_students) →
  (adjacent_female_students = 2) →
  (num_arrangements = 432) := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l3931_393191


namespace NUMINAMATH_CALUDE_union_problem_l3931_393107

def M : Set ℕ := {0, 1, 2}
def N (x : ℕ) : Set ℕ := {x}

theorem union_problem (x : ℕ) : M ∪ N x = {0, 1, 2, 3} → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_problem_l3931_393107


namespace NUMINAMATH_CALUDE_emily_beads_count_l3931_393120

/-- The number of necklaces Emily made -/
def necklaces : ℕ := 26

/-- The number of beads required for each necklace -/
def beads_per_necklace : ℕ := 2

/-- The total number of beads Emily had -/
def total_beads : ℕ := necklaces * beads_per_necklace

/-- Theorem stating that the total number of beads Emily had is 52 -/
theorem emily_beads_count : total_beads = 52 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_count_l3931_393120


namespace NUMINAMATH_CALUDE_raine_steps_theorem_l3931_393195

/-- The number of steps Raine takes to walk to school -/
def steps_to_school : ℕ := 150

/-- The number of days Raine walks to and from school -/
def days : ℕ := 5

/-- The total number of steps Raine takes in the given number of days -/
def total_steps : ℕ := 2 * steps_to_school * days

theorem raine_steps_theorem : total_steps = 1500 := by
  sorry

end NUMINAMATH_CALUDE_raine_steps_theorem_l3931_393195


namespace NUMINAMATH_CALUDE_watertown_marching_band_max_size_l3931_393167

theorem watertown_marching_band_max_size :
  ∀ n : ℕ,
  (25 * n < 1200) →
  (25 * n % 29 = 6) →
  (∀ m : ℕ, (25 * m < 1200) → (25 * m % 29 = 6) → m ≤ n) →
  25 * n = 1050 :=
by sorry

end NUMINAMATH_CALUDE_watertown_marching_band_max_size_l3931_393167


namespace NUMINAMATH_CALUDE_village_population_equality_l3931_393172

/-- The initial population of Village X -/
def initial_population_X : ℕ := 78000

/-- The yearly decrease in population of Village X -/
def decrease_rate_X : ℕ := 1200

/-- The yearly increase in population of Village Y -/
def increase_rate_Y : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years : ℕ := 18

/-- The initial population of Village Y -/
def initial_population_Y : ℕ := 42000

theorem village_population_equality :
  initial_population_X - decrease_rate_X * years = 
  initial_population_Y + increase_rate_Y * years :=
by sorry

#check village_population_equality

end NUMINAMATH_CALUDE_village_population_equality_l3931_393172


namespace NUMINAMATH_CALUDE_fourth_jeweler_bags_l3931_393108

def bags : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def total_gold : ℕ := bags.sum

theorem fourth_jeweler_bags (lost_bag : ℕ) 
  (h1 : lost_bag ∈ bags)
  (h2 : lost_bag ≠ 1 ∧ lost_bag ≠ 3 ∧ lost_bag ≠ 11)
  (h3 : (total_gold - lost_bag) % 4 = 0)
  (h4 : (bags.length - 1) % 4 = 0) :
  ∃ (jeweler1 jeweler2 jeweler3 jeweler4 : List ℕ),
    jeweler1.sum = jeweler2.sum ∧
    jeweler2.sum = jeweler3.sum ∧
    jeweler3.sum = jeweler4.sum ∧
    jeweler1.length = jeweler2.length ∧
    jeweler2.length = jeweler3.length ∧
    jeweler3.length = jeweler4.length ∧
    1 ∈ jeweler1 ∧
    3 ∈ jeweler2 ∧
    11 ∈ jeweler3 ∧
    jeweler4 = [2, 9, 10] :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_jeweler_bags_l3931_393108


namespace NUMINAMATH_CALUDE_root_magnitude_l3931_393168

theorem root_magnitude (a b : ℝ) (z : ℂ) (h : z = 1 + b * Complex.I) 
  (h_root : z ^ 2 + a * z + 3 = 0) : Complex.abs z = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_root_magnitude_l3931_393168


namespace NUMINAMATH_CALUDE_distance_to_leg_intersection_l3931_393160

/-- An isosceles trapezoid with specific diagonal properties -/
structure IsoscelesTrapezoid where
  /-- The length of the longer segment of each diagonal -/
  long_segment : ℝ
  /-- The length of the shorter segment of each diagonal -/
  short_segment : ℝ
  /-- The angle between the diagonals formed by the legs -/
  diagonal_angle : ℝ
  /-- Condition: The longer segment is 7 -/
  long_is_7 : long_segment = 7
  /-- Condition: The shorter segment is 3 -/
  short_is_3 : short_segment = 3
  /-- Condition: The angle between diagonals is 60° -/
  angle_is_60 : diagonal_angle = 60

/-- The theorem stating the distance from diagonal intersection to leg intersection -/
theorem distance_to_leg_intersection (t : IsoscelesTrapezoid) :
  (t.long_segment / t.short_segment) * t.short_segment = 21 / 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_leg_intersection_l3931_393160


namespace NUMINAMATH_CALUDE_decimal_51_to_binary_l3931_393133

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Checks if a list of booleans represents the binary form of a given natural number -/
def is_binary_of (bits : List Bool) (n : ℕ) : Prop :=
  to_binary n = bits.reverse

theorem decimal_51_to_binary :
  is_binary_of [true, true, false, false, true, true] 51 := by
  sorry

end NUMINAMATH_CALUDE_decimal_51_to_binary_l3931_393133


namespace NUMINAMATH_CALUDE_sequence_length_divisible_by_three_l3931_393192

theorem sequence_length_divisible_by_three (n : ℕ) (a : ℕ → ℝ) 
  (h1 : n ≥ 3)
  (h2 : ∀ i, a (i + n) = a i)
  (h3 : ∀ i, a i * a (i + 1) + 1 = a (i + 2)) :
  ∃ k : ℕ, n = 3 * k :=
sorry

end NUMINAMATH_CALUDE_sequence_length_divisible_by_three_l3931_393192


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3931_393194

/-- Given a rectangle with perimeter 72 meters and length-to-width ratio of 5:2,
    prove that its diagonal length is 194/7 meters. -/
theorem rectangle_diagonal (l w : ℝ) : 
  (2 * (l + w) = 72) →  -- perimeter condition
  (l = 5/2 * w) →       -- ratio condition
  Real.sqrt (l^2 + w^2) = 194/7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3931_393194


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l3931_393156

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l3931_393156


namespace NUMINAMATH_CALUDE_division_problem_l3931_393159

theorem division_problem : 
  (-1/42) / (1/6 - 3/14 + 2/3 - 2/7) = -1/14 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3931_393159


namespace NUMINAMATH_CALUDE_inequality_proof_l3931_393152

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a^4 + b^4 + c^4 - 2*(a^2*b^2 + a^2*c^2 + b^2*c^2) + a^2*b*c + b^2*a*c + c^2*a*b ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3931_393152


namespace NUMINAMATH_CALUDE_roots_product_theorem_l3931_393129

theorem roots_product_theorem (α β : ℝ) : 
  (α^2 + 2017*α + 1 = 0) → 
  (β^2 + 2017*β + 1 = 0) → 
  (1 + 2020*α + α^2) * (1 + 2020*β + β^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_theorem_l3931_393129


namespace NUMINAMATH_CALUDE_exists_coverable_prism_l3931_393104

/-- Represents a regular triangular prism -/
structure RegularTriangularPrism where
  base_side : ℝ
  lateral_edge : ℝ
  lateral_edge_eq : lateral_edge = Real.sqrt 3 * base_side

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ

/-- Predicate to check if a prism can be covered by equilateral triangles -/
def can_cover_with_equilateral_triangles (p : RegularTriangularPrism) (t : EquilateralTriangle) : Prop :=
  p.base_side = t.side ∧
  p.lateral_edge = Real.sqrt 3 * t.side

/-- Theorem stating that there exists a regular triangular prism that can be covered by equilateral triangles -/
theorem exists_coverable_prism : 
  ∃ (p : RegularTriangularPrism) (t : EquilateralTriangle), 
    can_cover_with_equilateral_triangles p t := by
  sorry

end NUMINAMATH_CALUDE_exists_coverable_prism_l3931_393104


namespace NUMINAMATH_CALUDE_cuboid_specific_surface_area_l3931_393143

/-- The surface area of a cuboid with given dimensions -/
def cuboid_surface_area (length breadth height : ℝ) : ℝ :=
  2 * (length * height + length * breadth + breadth * height)

/-- Theorem: The surface area of a cuboid with length 12 cm, breadth 6 cm, and height 10 cm is 504 cm² -/
theorem cuboid_specific_surface_area :
  cuboid_surface_area 12 6 10 = 504 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_specific_surface_area_l3931_393143


namespace NUMINAMATH_CALUDE_sum_of_mod2_and_mod3_l3931_393157

theorem sum_of_mod2_and_mod3 (a b : ℤ) : 
  a % 4 = 2 → b % 4 = 3 → (a + b) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_mod2_and_mod3_l3931_393157


namespace NUMINAMATH_CALUDE_remaining_digits_average_l3931_393141

theorem remaining_digits_average (total : ℕ) (subset : ℕ) (total_avg : ℚ) (subset_avg : ℚ) :
  total = 20 →
  subset = 14 →
  total_avg = 500 →
  subset_avg = 390 →
  let remaining := total - subset
  let remaining_sum := total * total_avg - subset * subset_avg
  remaining_sum / remaining = 756.67 := by
  sorry

end NUMINAMATH_CALUDE_remaining_digits_average_l3931_393141


namespace NUMINAMATH_CALUDE_sqrt_three_squared_equals_three_l3931_393175

theorem sqrt_three_squared_equals_three : (Real.sqrt 3)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_equals_three_l3931_393175


namespace NUMINAMATH_CALUDE_cube_sum_root_l3931_393132

theorem cube_sum_root : Real.sqrt (3^3 + 3^3 + 3^3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_root_l3931_393132


namespace NUMINAMATH_CALUDE_grid_coloring_theorem_l3931_393139

theorem grid_coloring_theorem (n : ℕ) :
  (∀ (grid : Fin 25 → Fin n → Fin 8),
    ∃ (cols : Fin 4 → Fin n) (rows : Fin 4 → Fin 25),
      ∀ (i j : Fin 4), grid (rows i) (cols j) = grid (rows 0) (cols 0)) ↔
  n ≥ 303601 :=
by sorry

end NUMINAMATH_CALUDE_grid_coloring_theorem_l3931_393139


namespace NUMINAMATH_CALUDE_sum_specific_repeating_decimals_l3931_393148

/-- Represents a repeating decimal with a whole number part and a repeating part -/
structure RepeatingDecimal where
  whole : ℕ
  repeating : ℕ
  base : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (d : RepeatingDecimal) : ℚ :=
  d.whole + (d.repeating : ℚ) / (d.base - 1 : ℚ)

/-- The sum of specific repeating decimals equals 3948/9999 -/
theorem sum_specific_repeating_decimals :
  let d1 := RepeatingDecimal.toRational ⟨0, 3, 10⟩
  let d2 := RepeatingDecimal.toRational ⟨0, 6, 100⟩
  let d3 := RepeatingDecimal.toRational ⟨0, 9, 10000⟩
  d1 + d2 + d3 = 3948 / 9999 := by
  sorry

#eval (3948 : ℚ) / 9999

end NUMINAMATH_CALUDE_sum_specific_repeating_decimals_l3931_393148


namespace NUMINAMATH_CALUDE_exists_square_with_2018_l3931_393161

theorem exists_square_with_2018 : ∃ n : ℕ, ∃ a b : ℕ, n^2 = a * 10000 + 2018 * b :=
  sorry

end NUMINAMATH_CALUDE_exists_square_with_2018_l3931_393161


namespace NUMINAMATH_CALUDE_sum_of_ages_in_5_years_l3931_393128

-- Define the current ages
def will_current_age : ℕ := 7
def diane_current_age : ℕ := 2 * will_current_age
def janet_current_age : ℕ := diane_current_age + 3

-- Define the ages in 5 years
def will_future_age : ℕ := will_current_age + 5
def diane_future_age : ℕ := diane_current_age + 5
def janet_future_age : ℕ := janet_current_age + 5

-- Theorem to prove
theorem sum_of_ages_in_5_years :
  will_future_age + diane_future_age + janet_future_age = 53 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_ages_in_5_years_l3931_393128


namespace NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l3931_393198

theorem negation_of_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l3931_393198


namespace NUMINAMATH_CALUDE_binary_to_decimal_octal_conversion_l3931_393196

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_octal (n : ℕ) : List ℕ :=
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_to_decimal_octal_conversion :
  (binary_to_decimal binary_101101 = 45) ∧
  (decimal_to_octal 45 = [5, 5]) := by
sorry

end NUMINAMATH_CALUDE_binary_to_decimal_octal_conversion_l3931_393196


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l3931_393140

def A : Nat := 111111
def B : Nat := 142857
def M : Nat := 1000000

theorem multiplicative_inverse_modulo :
  (63 * (A * B)) % M = 1 := by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l3931_393140


namespace NUMINAMATH_CALUDE_cody_money_calculation_l3931_393142

def final_money (initial : ℝ) (birthday_gift : ℝ) (game_cost : ℝ) (clothes_percentage : ℝ) (late_gift : ℝ) : ℝ :=
  let after_birthday := initial + birthday_gift
  let after_game := after_birthday - game_cost
  let clothes_cost := clothes_percentage * after_game
  let after_clothes := after_game - clothes_cost
  after_clothes + late_gift

theorem cody_money_calculation :
  final_money 45 9 19 0.4 4.5 = 25.5 := by
  sorry

end NUMINAMATH_CALUDE_cody_money_calculation_l3931_393142


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3931_393184

/-- Given a rectangular solid with length l, width w, and height h, 
    prove that if it satisfies certain volume change conditions, 
    its surface area is 290 square cm. -/
theorem rectangular_solid_surface_area 
  (l w h : ℝ) 
  (h1 : (l - 2) * w * h = l * w * h - 48)
  (h2 : l * (w + 3) * h = l * w * h + 99)
  (h3 : l * w * (h + 4) = l * w * h + 352)
  : 2 * (l * w + l * h + w * h) = 290 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3931_393184


namespace NUMINAMATH_CALUDE_group_size_proof_l3931_393190

/-- The number of people in the group -/
def n : ℕ := 10

/-- The weight increase of the group when the new person joins -/
def weight_increase : ℕ := 40

/-- The weight of the person being replaced -/
def old_weight : ℕ := 70

/-- The weight of the new person joining the group -/
def new_weight : ℕ := 110

/-- The average weight increase per person -/
def avg_increase : ℕ := 4

theorem group_size_proof :
  n * old_weight + weight_increase = n * (old_weight + avg_increase) :=
by sorry

end NUMINAMATH_CALUDE_group_size_proof_l3931_393190


namespace NUMINAMATH_CALUDE_exists_valid_assignment_l3931_393136

/-- Represents a rectangular parallelepiped with dimensions a, b, and c -/
structure Parallelepiped where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents an assignment of numbers to the unit squares on the surface of a parallelepiped -/
def SurfaceAssignment (p : Parallelepiped) := (ℕ × ℕ × ℕ) → ℝ

/-- Calculates the sum of numbers in a 1-width band surrounding the parallelepiped -/
def bandSum (p : Parallelepiped) (assignment : SurfaceAssignment p) : ℝ := sorry

/-- Theorem stating the existence of a valid assignment for a 3 × 4 × 5 parallelepiped -/
theorem exists_valid_assignment :
  ∃ (assignment : SurfaceAssignment ⟨3, 4, 5⟩),
    bandSum ⟨3, 4, 5⟩ assignment = 120 := by sorry

end NUMINAMATH_CALUDE_exists_valid_assignment_l3931_393136


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3931_393155

/-- Represents a hyperbola -/
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_axes : Bool
  eccentricity : ℝ

/-- The equation of asymptotes for a hyperbola -/
def asymptote_equation (h : Hyperbola) : Set (ℝ × ℝ) :=
  {(x, y) | y = x ∨ y = -x}

theorem hyperbola_asymptotes (C : Hyperbola) 
  (h1 : C.center = (0, 0)) 
  (h2 : C.foci_on_axes = true) 
  (h3 : C.eccentricity = Real.sqrt 2) : 
  asymptote_equation C = {(x, y) | y = x ∨ y = -x} := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3931_393155


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l3931_393170

theorem largest_solution_of_equation (a : ℝ) : 
  (3 * a + 4) * (a - 2) = 8 * a → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l3931_393170


namespace NUMINAMATH_CALUDE_coin_pile_theorem_l3931_393193

/-- Represents the state of the three piles of coins -/
structure CoinState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Defines the allowed operations on the coin piles -/
inductive Operation
  | split_even : ℕ → Operation
  | remove_odd : ℕ → Operation

/-- Applies an operation to a CoinState -/
def apply_operation (state : CoinState) (op : Operation) : CoinState :=
  sorry

/-- Checks if a state has a pile with at least 2017^2017 coins -/
def has_large_pile (state : CoinState) : Prop :=
  sorry

/-- Theorem stating that any initial state except (2015, 2015, 2015) can reach a large pile -/
theorem coin_pile_theorem (initial : CoinState)
    (h1 : initial.a ≥ 2015)
    (h2 : initial.b ≥ 2015)
    (h3 : initial.c ≥ 2015)
    (h4 : ¬(initial.a = 2015 ∧ initial.b = 2015 ∧ initial.c = 2015)) :
    ∃ (ops : List Operation), has_large_pile (ops.foldl apply_operation initial) :=
  sorry

end NUMINAMATH_CALUDE_coin_pile_theorem_l3931_393193


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3931_393164

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∃ d, ∀ n, a (n + 1) = a n + d)

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  2 * a 6 + 2 * a 8 = (a 7) ^ 2 → a 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3931_393164


namespace NUMINAMATH_CALUDE_cylinder_volume_equality_l3931_393180

/-- Given two identical cylinders with initial radius 5 inches and height 4 inches,
    prove that when the radius of one cylinder is increased by x inches and
    the height of the second cylinder is increased by 2 inches, resulting in
    equal volumes, x = (5(√6 - 2)) / 2. -/
theorem cylinder_volume_equality (x : ℝ) : 
  (π * (5 + x)^2 * 4 = π * 5^2 * (4 + 2)) → x = (5 * (Real.sqrt 6 - 2)) / 2 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_equality_l3931_393180


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3931_393153

/-- Given a right triangle ABC with angle C = 90°, side a = 12, and side b = 16, prove that the length of side c is 20. -/
theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 12 → b = 16 → c^2 = a^2 + b^2 → c = 20 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3931_393153


namespace NUMINAMATH_CALUDE_trivia_game_base_points_l3931_393163

/-- Calculates the base points per round for a trivia game -/
theorem trivia_game_base_points
  (total_rounds : ℕ)
  (total_score : ℕ)
  (bonus_points : ℕ)
  (penalty_points : ℕ)
  (h1 : total_rounds = 5)
  (h2 : total_score = 370)
  (h3 : bonus_points = 50)
  (h4 : penalty_points = 30) :
  (total_score + bonus_points - penalty_points) / total_rounds = 78 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_base_points_l3931_393163


namespace NUMINAMATH_CALUDE_product_greater_than_sum_l3931_393114

theorem product_greater_than_sum (a b : ℝ) (ha : a ≥ 2) (hb : b > 2) : a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_l3931_393114


namespace NUMINAMATH_CALUDE_exists_non_convex_polyhedron_with_no_visible_vertices_l3931_393146

/-- A polyhedron in 3D space -/
structure Polyhedron where
  vertices : Set (Fin 3 → ℝ)
  faces : Set (Set (Fin 3 → ℝ))
  is_valid : True  -- Additional conditions for a valid polyhedron would go here

/-- A point in 3D space -/
def Point3D := Fin 3 → ℝ

/-- Predicate to check if a polyhedron is non-convex -/
def is_non_convex (P : Polyhedron) : Prop := sorry

/-- Predicate to check if a point is outside a polyhedron -/
def is_outside (M : Point3D) (P : Polyhedron) : Prop := sorry

/-- Predicate to check if a vertex is visible from a point -/
def is_visible (v : Point3D) (M : Point3D) (P : Polyhedron) : Prop := sorry

/-- Theorem stating the existence of a non-convex polyhedron with no visible vertices from an exterior point -/
theorem exists_non_convex_polyhedron_with_no_visible_vertices :
  ∃ (P : Polyhedron) (M : Point3D),
    is_non_convex P ∧
    is_outside M P ∧
    ∀ v ∈ P.vertices, ¬is_visible v M P := by
  sorry

end NUMINAMATH_CALUDE_exists_non_convex_polyhedron_with_no_visible_vertices_l3931_393146


namespace NUMINAMATH_CALUDE_ray_AB_bisects_PAQ_l3931_393174

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 5/2)^2 = 25/4

-- Define points A and B on the y-axis
def point_A : ℝ × ℝ := (0, 4)
def point_B : ℝ × ℝ := (0, 1)

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2/8 + y^2/4 = 1

-- Define line l passing through B
def line_l (x y : ℝ) (k : ℝ) : Prop :=
  y = k * x + 1

-- Theorem statement
theorem ray_AB_bisects_PAQ :
  ∀ (P Q : ℝ × ℝ) (k : ℝ),
    circle_C 2 0 →  -- Circle C is tangent to x-axis at T(2,0)
    |point_A.2 - point_B.2| = 3 →  -- |AB| = 3
    line_l P.1 P.2 k →  -- P is on line l
    line_l Q.1 Q.2 k →  -- Q is on line l
    ellipse P.1 P.2 →  -- P is on the ellipse
    ellipse Q.1 Q.2 →  -- Q is on the ellipse
    -- Ray AB bisects angle PAQ
    (P.2 - point_A.2) / (P.1 - point_A.1) + (Q.2 - point_A.2) / (Q.1 - point_A.1) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ray_AB_bisects_PAQ_l3931_393174


namespace NUMINAMATH_CALUDE_partition_condition_l3931_393126

theorem partition_condition (α β : ℕ+) : 
  (∃ (A B : Set ℕ+), 
    (A ∪ B = Set.univ) ∧ 
    (A ∩ B = ∅) ∧ 
    ({α * a | a ∈ A} = {β * b | b ∈ B})) ↔ 
  (α ∣ β ∧ α ≠ β) ∨ (β ∣ α ∧ α ≠ β) :=
sorry

end NUMINAMATH_CALUDE_partition_condition_l3931_393126


namespace NUMINAMATH_CALUDE_equation_solution_l3931_393185

theorem equation_solution :
  ∃ x : ℝ, (10 : ℝ)^x * 500^x = 1000000^3 ∧ x = 18 / 3.699 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3931_393185


namespace NUMINAMATH_CALUDE_alexa_katerina_weight_l3931_393138

/-- The combined weight of Alexa and Katerina is 92 pounds -/
theorem alexa_katerina_weight (total_weight : ℕ) (alexa_weight : ℕ) (michael_weight : ℕ)
  (h1 : total_weight = 154)
  (h2 : alexa_weight = 46)
  (h3 : michael_weight = 62) :
  total_weight - michael_weight = 92 :=
by sorry

end NUMINAMATH_CALUDE_alexa_katerina_weight_l3931_393138


namespace NUMINAMATH_CALUDE_median_length_in_right_triangle_l3931_393119

theorem median_length_in_right_triangle (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let c := Real.sqrt (a^2 + b^2)  -- hypotenuse
  let m := Real.sqrt (b^2 + (a/2)^2)  -- median
  (∃ k : ℝ, k = 0.51 ∧ m = k * c) ∧ ¬(∃ k : ℝ, k = 0.49 ∧ m = k * c) :=
by sorry

end NUMINAMATH_CALUDE_median_length_in_right_triangle_l3931_393119


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3931_393145

/-- The number of ways to place n distinguishable balls into k indistinguishable boxes -/
def ball_distribution (n k : ℕ) : ℕ := sorry

/-- The number of ways to place 5 distinguishable balls into 3 indistinguishable boxes is 36 -/
theorem five_balls_three_boxes : ball_distribution 5 3 = 36 := by sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3931_393145


namespace NUMINAMATH_CALUDE_thabo_hardcover_count_l3931_393186

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (bc : BookCollection) : Prop :=
  bc.hardcover_nonfiction + bc.paperback_nonfiction + bc.paperback_fiction = 160 ∧
  bc.paperback_nonfiction = bc.hardcover_nonfiction + 20 ∧
  bc.paperback_fiction = 2 * bc.paperback_nonfiction

theorem thabo_hardcover_count (bc : BookCollection) 
  (h : is_valid_collection bc) : bc.hardcover_nonfiction = 25 := by
  sorry

end NUMINAMATH_CALUDE_thabo_hardcover_count_l3931_393186


namespace NUMINAMATH_CALUDE_sine_function_value_l3931_393199

/-- Proves that f(π/4) = -4/5 given specific conditions on φ and ω -/
theorem sine_function_value (φ ω : Real) (h1 : (-4 : Real) / 5 = Real.cos φ)
    (h2 : (3 : Real) / 5 = Real.sin φ) (h3 : ω > 0) 
    (h4 : π / (2 * ω) = π / 2) : 
  Real.sin (ω * (π / 4) + φ) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_value_l3931_393199


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3931_393176

theorem quadratic_inequality_solution_set (a : ℝ) : 
  (∀ x, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) → 
  (∀ x, ax^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3931_393176


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l3931_393101

theorem polynomial_product_sum (x : ℝ) : ∃ (a b c d e : ℝ),
  (2 * x^3 - 3 * x^2 + 5 * x - 1) * (8 - 3 * x) = 
    a * x^4 + b * x^3 + c * x^2 + d * x + e ∧
  16 * a + 8 * b + 4 * c + 2 * d + e = 26 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l3931_393101


namespace NUMINAMATH_CALUDE_backpack_price_relation_l3931_393100

theorem backpack_price_relation (x : ℝ) : x > 0 →
  (810 : ℝ) / (x + 20) = (600 : ℝ) / x * (1 - 0.1) := by
  sorry

end NUMINAMATH_CALUDE_backpack_price_relation_l3931_393100


namespace NUMINAMATH_CALUDE_triangle_shape_l3931_393102

theorem triangle_shape (a b c : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h3 : a^2 * c^2 + b^2 * c^2 = a^4 - b^4) : 
  a^2 = b^2 + c^2 := by sorry

end NUMINAMATH_CALUDE_triangle_shape_l3931_393102


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l3931_393135

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) → 
  -1 < a ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l3931_393135


namespace NUMINAMATH_CALUDE_soccer_team_winning_percentage_l3931_393131

/-- Calculates the winning percentage of a soccer team -/
def winning_percentage (games_played : ℕ) (games_won : ℕ) : ℚ :=
  (games_won : ℚ) / (games_played : ℚ) * 100

/-- Theorem stating that a team with 280 games played and 182 wins has a 65% winning percentage -/
theorem soccer_team_winning_percentage :
  let games_played : ℕ := 280
  let games_won : ℕ := 182
  winning_percentage games_played games_won = 65 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_winning_percentage_l3931_393131


namespace NUMINAMATH_CALUDE_matrix_vector_product_plus_vector_l3931_393134

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; -5, 6]
def v : Matrix (Fin 2) (Fin 1) ℝ := !![5; -2]
def w : Matrix (Fin 2) (Fin 1) ℝ := !![1; -1]

theorem matrix_vector_product_plus_vector :
  A * v + w = !![25; -38] := by sorry

end NUMINAMATH_CALUDE_matrix_vector_product_plus_vector_l3931_393134


namespace NUMINAMATH_CALUDE_minimum_score_to_increase_average_l3931_393154

def scores : List ℕ := [84, 76, 89, 94, 67, 90]

def current_average : ℚ := (scores.sum : ℚ) / scores.length

def target_average : ℚ := current_average + 5

def required_score : ℕ := 118

theorem minimum_score_to_increase_average : 
  (((scores.sum + required_score : ℚ) / (scores.length + 1)) = target_average) ∧
  (∀ (s : ℕ), s < required_score → 
    ((scores.sum + s : ℚ) / (scores.length + 1)) < target_average) := by
  sorry

end NUMINAMATH_CALUDE_minimum_score_to_increase_average_l3931_393154


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3931_393151

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem cubic_function_properties :
  ∀ (a b c : ℝ),
  (∀ x : ℝ, f a b c x ≤ f a b c (-1)) ∧  -- Maximum at x = -1
  (f a b c (-1) = 7) ∧                   -- Maximum value is 7
  (∀ x : ℝ, f a b c x ≥ f a b c 3) →     -- Minimum at x = 3
  (a = -3 ∧ b = -9 ∧ c = 2 ∧ f a b c 3 = -25) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3931_393151


namespace NUMINAMATH_CALUDE_max_area_at_45_degrees_l3931_393106

/-- A screen in a room corner --/
structure Screen where
  length : ℝ
  angle : ℝ

/-- Configuration of two screens in a room corner --/
structure CornerScreens where
  screen1 : Screen
  screen2 : Screen

/-- The area enclosed by two screens in a room corner --/
noncomputable def enclosedArea (cs : CornerScreens) : ℝ := sorry

/-- Theorem: The area enclosed by two equal-length screens in a right-angled corner
    is maximized when each screen forms a 45° angle with its adjacent wall --/
theorem max_area_at_45_degrees (l : ℝ) (h : l > 0) :
  ∃ (cs : CornerScreens),
    cs.screen1.length = l ∧
    cs.screen2.length = l ∧
    cs.screen1.angle = π/4 ∧
    cs.screen2.angle = π/4 ∧
    ∀ (other : CornerScreens),
      other.screen1.length = l →
      other.screen2.length = l →
      enclosedArea other ≤ enclosedArea cs :=
sorry

end NUMINAMATH_CALUDE_max_area_at_45_degrees_l3931_393106


namespace NUMINAMATH_CALUDE_right_triangle_increased_sides_is_acute_l3931_393171

/-- 
Given a right-angled triangle with sides a, b, and c (where c is the hypotenuse),
and a positive real number d, prove that the triangle with sides (a+d), (b+d), and (c+d)
is an acute-angled triangle.
-/
theorem right_triangle_increased_sides_is_acute 
  (a b c d : ℝ) 
  (h_right : a^2 + b^2 = c^2)  -- Original triangle is right-angled
  (h_pos : d > 0)              -- Increase is positive
  : (a+d)^2 + (b+d)^2 > (c+d)^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_increased_sides_is_acute_l3931_393171


namespace NUMINAMATH_CALUDE_christen_peeled_twenty_potatoes_l3931_393147

/-- Calculates the number of potatoes Christen peeled -/
def christenPotatoesPeeled (initialPotatoes : ℕ) (homerRate : ℕ) (christenRate : ℕ) (timeBeforeJoining : ℕ) : ℕ :=
  let potatoesLeftWhenChristenJoins := initialPotatoes - homerRate * timeBeforeJoining
  let combinedRate := homerRate + christenRate
  let timeToFinish := potatoesLeftWhenChristenJoins / combinedRate
  christenRate * timeToFinish

/-- Theorem stating that Christen peeled 20 potatoes given the initial conditions -/
theorem christen_peeled_twenty_potatoes :
  christenPotatoesPeeled 44 3 5 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_twenty_potatoes_l3931_393147


namespace NUMINAMATH_CALUDE_bridget_bakery_profit_l3931_393110

/-- Calculates the profit for Bridget's bakery given the specified conditions. -/
def bakery_profit (total_loaves : ℕ) (morning_price afternoon_price late_price : ℚ)
  (operational_cost production_cost : ℚ) : ℚ :=
  let morning_sales := (2 : ℚ) / 5 * total_loaves
  let afternoon_sales := (1 : ℚ) / 2 * (total_loaves - morning_sales)
  let late_sales := (2 : ℚ) / 3 * (total_loaves - morning_sales - afternoon_sales)
  
  let revenue := morning_sales * morning_price + 
                 afternoon_sales * afternoon_price + 
                 late_sales * late_price
  
  let cost := (total_loaves : ℚ) * production_cost + operational_cost
  
  revenue - cost

/-- Theorem stating that under the given conditions, Bridget's bakery profit is $53. -/
theorem bridget_bakery_profit :
  bakery_profit 60 3 (3/2) 2 10 1 = 53 := by
  sorry

#eval bakery_profit 60 3 (3/2) 2 10 1

end NUMINAMATH_CALUDE_bridget_bakery_profit_l3931_393110


namespace NUMINAMATH_CALUDE_aarons_age_l3931_393127

theorem aarons_age (aaron julie : ℕ) 
  (h1 : julie = 4 * aaron)
  (h2 : julie + 10 = 2 * (aaron + 10)) :
  aaron = 5 := by
sorry

end NUMINAMATH_CALUDE_aarons_age_l3931_393127


namespace NUMINAMATH_CALUDE_tommys_tomato_profit_l3931_393197

/-- Represents the problem of calculating Tommy's profit from selling tomatoes --/
theorem tommys_tomato_profit :
  let crate_capacity : ℕ := 20  -- kg
  let num_crates : ℕ := 3
  let purchase_cost : ℕ := 330  -- $
  let selling_price : ℕ := 6    -- $ per kg
  let rotten_tomatoes : ℕ := 3  -- kg
  
  let total_tomatoes : ℕ := crate_capacity * num_crates
  let sellable_tomatoes : ℕ := total_tomatoes - rotten_tomatoes
  let revenue : ℕ := sellable_tomatoes * selling_price
  let profit : ℤ := revenue - purchase_cost
  
  profit = 12 := by sorry

end NUMINAMATH_CALUDE_tommys_tomato_profit_l3931_393197


namespace NUMINAMATH_CALUDE_a_divisibility_characterization_l3931_393125

/-- Sequence a_n defined recursively -/
def a : ℕ → ℤ
  | 0 => 3
  | 1 => 9
  | (n + 2) => 4 * a (n + 1) - 3 * a n - 4 * (n + 2) + 2

/-- Predicate for n such that a_n is divisible by 9 -/
def is_divisible_by_9 (n : ℕ) : Prop :=
  n = 1 ∨ n % 9 = 7 ∨ n % 9 = 8

theorem a_divisibility_characterization :
  ∀ n : ℕ, 9 ∣ a n ↔ is_divisible_by_9 n :=
sorry

end NUMINAMATH_CALUDE_a_divisibility_characterization_l3931_393125


namespace NUMINAMATH_CALUDE_f_of_two_equals_eleven_l3931_393112

/-- A function f satisfying the given conditions -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b * x + 3

/-- The theorem stating that f(2) = 11 under the given conditions -/
theorem f_of_two_equals_eleven (a b : ℝ) 
  (h1 : f a b 1 = 7) 
  (h3 : f a b 3 = 15) : 
  f a b 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_equals_eleven_l3931_393112


namespace NUMINAMATH_CALUDE_computer_multiplications_l3931_393137

/-- Represents the number of multiplications a computer can perform per second -/
def multiplications_per_second : ℕ := 15000

/-- Represents the number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Represents the number of hours we're calculating for -/
def hours : ℕ := 2

/-- Theorem stating that the computer will perform 108 million multiplications in two hours -/
theorem computer_multiplications :
  multiplications_per_second * seconds_per_hour * hours = 108000000 := by
  sorry

#eval multiplications_per_second * seconds_per_hour * hours

end NUMINAMATH_CALUDE_computer_multiplications_l3931_393137


namespace NUMINAMATH_CALUDE_prob_four_or_full_house_after_reroll_l3931_393149

-- Define a die as having 6 sides
def die_sides : ℕ := 6

-- Define the number of dice
def num_dice : ℕ := 6

-- Define a function to represent the probability of a specific outcome when rolling a die
def prob_specific_outcome (sides : ℕ) : ℚ := 1 / sides

-- Define the probability of getting a four-of-a-kind or a full house after re-rolling
def prob_four_or_full_house : ℚ := 1 / 3

-- State the theorem
theorem prob_four_or_full_house_after_reroll 
  (h1 : ∃ (triple pair : ℕ), triple ≠ pair ∧ triple ≤ die_sides ∧ pair ≤ die_sides) 
  (h2 : ¬ ∃ (four : ℕ), four ≤ die_sides) :
  prob_four_or_full_house = prob_specific_outcome die_sides + prob_specific_outcome die_sides :=
sorry

end NUMINAMATH_CALUDE_prob_four_or_full_house_after_reroll_l3931_393149


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3931_393177

theorem quadratic_equation_solution (a k : ℝ) : 
  (∃ x : ℝ, a * x^2 + 3 * x - k = 0) → 
  (k = 44) → 
  (a * 4^2 + 3 * 4 - k = 0) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3931_393177
