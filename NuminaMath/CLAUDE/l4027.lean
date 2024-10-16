import Mathlib

namespace NUMINAMATH_CALUDE_unique_three_digit_divisible_by_nine_l4027_402774

theorem unique_three_digit_divisible_by_nine : 
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit number
    n % 10 = 4 ∧          -- units digit is 4
    n / 100 = 6 ∧         -- hundreds digit is 6
    n % 9 = 0 ∧           -- divisible by 9
    n = 684               -- the number is 684
  := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_divisible_by_nine_l4027_402774


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l4027_402775

theorem trigonometric_expression_equality (x : Real) :
  (x > π / 2 ∧ x < π) →  -- x is in the second quadrant
  (Real.tan x)^2 + 3 * Real.tan x - 4 = 0 →
  (Real.sin x + Real.cos x) / (2 * Real.sin x - Real.cos x) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l4027_402775


namespace NUMINAMATH_CALUDE_first_digit_891_base8_l4027_402754

/-- Represents a positive integer in a given base --/
def BaseRepresentation (n : ℕ+) (base : ℕ) : List ℕ :=
  sorry

/-- Returns the first (leftmost) digit of a number's representation in a given base --/
def firstDigit (n : ℕ+) (base : ℕ) : ℕ :=
  match BaseRepresentation n base with
  | [] => 0  -- This case should never occur for positive integers
  | d::_ => d

theorem first_digit_891_base8 :
  firstDigit 891 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_891_base8_l4027_402754


namespace NUMINAMATH_CALUDE_mobile_phone_price_decrease_l4027_402792

theorem mobile_phone_price_decrease (current_price : ℝ) (yearly_decrease_rate : ℝ) (years : ℕ) : 
  current_price = 1000 ∧ yearly_decrease_rate = 0.2 ∧ years = 2 →
  current_price = (1562.5 : ℝ) * (1 - yearly_decrease_rate) ^ years := by
sorry

end NUMINAMATH_CALUDE_mobile_phone_price_decrease_l4027_402792


namespace NUMINAMATH_CALUDE_correct_expansion_l4027_402717

theorem correct_expansion (x : ℝ) : (-3*x + 2) * (-3*x - 2) = 9*x^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_expansion_l4027_402717


namespace NUMINAMATH_CALUDE_theme_parks_sum_l4027_402716

theorem theme_parks_sum (jamestown venice marina_del_ray : ℕ) : 
  jamestown = 20 →
  venice = jamestown + 25 →
  marina_del_ray = jamestown + 50 →
  jamestown + venice + marina_del_ray = 135 := by
  sorry

end NUMINAMATH_CALUDE_theme_parks_sum_l4027_402716


namespace NUMINAMATH_CALUDE_individual_contribution_proof_l4027_402793

def total_contribution : ℝ := 90
def class_funds : ℝ := 30
def num_students : ℝ := 25

theorem individual_contribution_proof :
  (total_contribution - class_funds) / num_students = 2.40 :=
by sorry

end NUMINAMATH_CALUDE_individual_contribution_proof_l4027_402793


namespace NUMINAMATH_CALUDE_debt_work_hours_l4027_402771

def initial_debt : ℝ := 100
def payment : ℝ := 40
def hourly_rate : ℝ := 15

theorem debt_work_hours : 
  (initial_debt - payment) / hourly_rate = 4 := by sorry

end NUMINAMATH_CALUDE_debt_work_hours_l4027_402771


namespace NUMINAMATH_CALUDE_eu_countries_2012_is_set_l4027_402746

/-- A type representing countries -/
def Country : Type := String

/-- A predicate that determines if a country was in the EU in 2012 -/
def WasEUMemberIn2012 (c : Country) : Prop := sorry

/-- The set of all EU countries in 2012 -/
def EUCountries2012 : Set Country :=
  {c : Country | WasEUMemberIn2012 c}

/-- A property that determines if a collection can form a set -/
def CanFormSet (S : Set α) : Prop :=
  ∀ x, x ∈ S → (∃ p : Prop, p ↔ x ∈ S)

theorem eu_countries_2012_is_set :
  CanFormSet EUCountries2012 :=
sorry

end NUMINAMATH_CALUDE_eu_countries_2012_is_set_l4027_402746


namespace NUMINAMATH_CALUDE_triangle_problem_l4027_402764

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_problem (abc : Triangle) 
  (h1 : abc.c = 13)
  (h2 : Real.cos abc.A = 5/13) :
  (abc.a = 36 → Real.sin abc.C = 1/3) ∧ 
  (abc.a * abc.b * Real.sin abc.C / 2 = 6 → abc.a = 4 * Real.sqrt 10 ∧ abc.b = 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l4027_402764


namespace NUMINAMATH_CALUDE_sector_arc_length_l4027_402767

/-- Given a sector with circumference 8 and central angle 2 radians, 
    the length of its arc is 4 -/
theorem sector_arc_length (c : ℝ) (θ : ℝ) (l : ℝ) (r : ℝ) : 
  c = 8 →  -- circumference of the sector
  θ = 2 →  -- central angle in radians
  c = l + 2 * r →  -- circumference formula for a sector
  l = r * θ →  -- arc length formula
  l = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l4027_402767


namespace NUMINAMATH_CALUDE_square_perimeter_l4027_402732

/-- A square with area 484 cm² has a perimeter of 88 cm. -/
theorem square_perimeter (s : ℝ) (h1 : s > 0) (h2 : s^2 = 484) : 4 * s = 88 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l4027_402732


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4027_402785

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 2 → (x + 1) * (x - 2) > 0) ∧
  (∃ x, (x + 1) * (x - 2) > 0 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4027_402785


namespace NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l4027_402733

/-- The ratio of the volume of water in a cone filled to 2/3 of its height to the total volume of the cone is 8/27. -/
theorem water_volume_ratio_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let water_height := (2 : ℝ) / 3 * h
  let water_radius := (2 : ℝ) / 3 * r
  let cone_volume := (1 : ℝ) / 3 * π * r^2 * h
  let water_volume := (1 : ℝ) / 3 * π * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 := by
sorry


end NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l4027_402733


namespace NUMINAMATH_CALUDE_sequence_properties_l4027_402700

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 3 * n - 1

-- Define the geometric sequence b_n
def b (n : ℕ) : ℚ := 2^n

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

theorem sequence_properties :
  (a 1 = 2) ∧
  (b 1 = 2) ∧
  (a 4 + b 4 = 27) ∧
  (S 4 - b 4 = 10) ∧
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) ∧
  (∀ n : ℕ, b (n + 1) / b n = b 2 / b 1) ∧
  (∃ m : ℕ+, 
    (4 / m : ℚ) * a (7 - m) = (b 1)^2 ∧
    (4 / m : ℚ) * a 7 = (b 2)^2 ∧
    (4 / m : ℚ) * a (7 + 4 * m) = (b 3)^2) :=
by sorry

#check sequence_properties

end NUMINAMATH_CALUDE_sequence_properties_l4027_402700


namespace NUMINAMATH_CALUDE_trigonometric_identities_l4027_402737

theorem trigonometric_identities (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : 3 * Real.sin α = 4 * Real.cos α)
  (h4 : Real.cos (α + β) = -(2 * Real.sqrt 5) / 5) :
  Real.cos (2 * α) = -7 / 25 ∧ Real.sin β = (2 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l4027_402737


namespace NUMINAMATH_CALUDE_cosine_equality_l4027_402789

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (830 * π / 180) → n = 70 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l4027_402789


namespace NUMINAMATH_CALUDE_credit_card_more_beneficial_l4027_402783

/-- Represents the purchase amount in rubles -/
def purchase_amount : ℝ := 20000

/-- Represents the credit card cashback rate -/
def credit_cashback_rate : ℝ := 0.005

/-- Represents the debit card cashback rate -/
def debit_cashback_rate : ℝ := 0.01

/-- Represents the annual interest rate on the debit card -/
def annual_interest_rate : ℝ := 0.06

/-- Represents the number of days in a year for interest calculation -/
def days_in_year : ℝ := 360

/-- Represents the minimum number of days for credit card to be more beneficial -/
def min_days : ℕ := 31

theorem credit_card_more_beneficial :
  ∀ N : ℕ,
  N ≥ min_days →
  (purchase_amount * credit_cashback_rate) + 
  (purchase_amount * annual_interest_rate * N / days_in_year) >
  purchase_amount * debit_cashback_rate :=
sorry

end NUMINAMATH_CALUDE_credit_card_more_beneficial_l4027_402783


namespace NUMINAMATH_CALUDE_monotonic_condition_l4027_402704

/-- A function f is monotonic on an interval [a, b] if it is either
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The main theorem stating the condition for the function to be monotonic. -/
theorem monotonic_condition (a : ℝ) :
  (IsMonotonic (fun x => -x^2 + 4*a*x) 2 4) ↔ (a ≤ 1 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_monotonic_condition_l4027_402704


namespace NUMINAMATH_CALUDE_equal_bill_time_l4027_402757

/-- United Telephone's base rate -/
def united_base : ℝ := 8

/-- United Telephone's per-minute rate -/
def united_per_minute : ℝ := 0.25

/-- Atlantic Call's base rate -/
def atlantic_base : ℝ := 12

/-- Atlantic Call's per-minute rate -/
def atlantic_per_minute : ℝ := 0.20

/-- The number of minutes at which the bills are equal -/
def equal_minutes : ℝ := 80

theorem equal_bill_time :
  united_base + united_per_minute * equal_minutes =
  atlantic_base + atlantic_per_minute * equal_minutes :=
by sorry

end NUMINAMATH_CALUDE_equal_bill_time_l4027_402757


namespace NUMINAMATH_CALUDE_gcd_40304_30213_l4027_402728

theorem gcd_40304_30213 : Nat.gcd 40304 30213 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_40304_30213_l4027_402728


namespace NUMINAMATH_CALUDE_f_properties_l4027_402710

noncomputable def f (x : ℝ) : ℝ := x * (Real.exp x + Real.exp (-x))

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4027_402710


namespace NUMINAMATH_CALUDE_sum_of_abs_values_l4027_402734

theorem sum_of_abs_values (a b : ℝ) (ha : |a| = 4) (hb : |b| = 5) :
  (a + b = 9) ∨ (a + b = -9) ∨ (a + b = 1) ∨ (a + b = -1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abs_values_l4027_402734


namespace NUMINAMATH_CALUDE_floor_equation_solution_l4027_402707

theorem floor_equation_solution (a b : ℕ+) :
  (Int.floor (a^2 / b : ℚ) + Int.floor (b^2 / a : ℚ) = 
   Int.floor ((a^2 + b^2) / (a * b) : ℚ) + a * b) ↔ 
  (a = b^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l4027_402707


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l4027_402766

-- Define the universe set U
def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}

-- Define set A
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x | 3 < x ∧ x ≤ 7}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 3 < x ∧ x < 5} :=
sorry

-- Theorem for the union of complement of A and B
theorem union_complement_A_B : (U \ A) ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 < x ∧ x ≤ 7)} :=
sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l4027_402766


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l4027_402738

/-- The shortest distance from a point on the parabola y = x^2 to the line x - y - 2 = 0 is 7√2/8 -/
theorem shortest_distance_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2}
  let line := {p : ℝ × ℝ | p.1 - p.2 - 2 = 0}
  ∃ d : ℝ, d = 7 * Real.sqrt 2 / 8 ∧
    ∀ p ∈ parabola, ∀ q ∈ line,
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l4027_402738


namespace NUMINAMATH_CALUDE_maryGarbageBillIs102_l4027_402763

/-- Calculates Mary's monthly garbage bill --/
def maryGarbageBill : ℚ :=
  let trashBinCharge : ℚ := 10
  let recyclingBinCharge : ℚ := 5
  let trashBinCount : ℕ := 2
  let recyclingBinCount : ℕ := 1
  let weeksInMonth : ℕ := 4
  let elderlyDiscountPercentage : ℚ := 18 / 100
  let inappropriateItemsFine : ℚ := 20

  let weeklyCharge := trashBinCharge * trashBinCount + recyclingBinCharge * recyclingBinCount
  let monthlyCharge := weeklyCharge * weeksInMonth
  let discountAmount := monthlyCharge * elderlyDiscountPercentage
  let discountedMonthlyCharge := monthlyCharge - discountAmount
  discountedMonthlyCharge + inappropriateItemsFine

theorem maryGarbageBillIs102 : maryGarbageBill = 102 := by
  sorry

end NUMINAMATH_CALUDE_maryGarbageBillIs102_l4027_402763


namespace NUMINAMATH_CALUDE_no_valid_chess_sequence_l4027_402740

/-- Represents a sequence of moves on a 6x6 chessboard -/
def ChessSequence := Fin 36 → Fin 36

/-- Checks if the difference between consecutive terms alternates between 1 and 2 -/
def validMoves (seq : ChessSequence) : Prop :=
  ∀ i : Fin 35, (i.val % 2 = 0 → |seq (i + 1) - seq i| = 1) ∧
                (i.val % 2 = 1 → |seq (i + 1) - seq i| = 2)

/-- Checks if all elements in the sequence are distinct -/
def allDistinct (seq : ChessSequence) : Prop :=
  ∀ i j : Fin 36, i ≠ j → seq i ≠ seq j

/-- The main theorem: no valid chess sequence exists -/
theorem no_valid_chess_sequence :
  ¬∃ (seq : ChessSequence), validMoves seq ∧ allDistinct seq :=
sorry

end NUMINAMATH_CALUDE_no_valid_chess_sequence_l4027_402740


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l4027_402747

theorem boys_to_girls_ratio : 
  ∀ (boys girls : ℕ), 
    boys = 40 →
    girls = boys + 64 →
    (boys : ℚ) / (girls : ℚ) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l4027_402747


namespace NUMINAMATH_CALUDE_no_unique_solution_l4027_402759

theorem no_unique_solution (x y z : ℕ+) : 
  ¬∃ (f : ℕ+ → ℕ+ → ℕ+ → Prop), 
    (∀ (a b c : ℕ+), f a b c ↔ Real.sqrt (a^2 + Real.sqrt ((b:ℝ)/(c:ℝ))) = (b:ℝ)^2 * Real.sqrt ((a:ℝ)/(c:ℝ))) ∧
    (∃! (g : ℕ+ → ℕ+ → ℕ+ → Prop), ∀ (a b c : ℕ+), g a b c ↔ f a b c) :=
sorry

end NUMINAMATH_CALUDE_no_unique_solution_l4027_402759


namespace NUMINAMATH_CALUDE_colored_plane_triangles_l4027_402743

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point
  sideLength : ℝ
  isEquilateral : 
    (a.x - b.x)^2 + (a.y - b.y)^2 = sideLength^2 ∧
    (b.x - c.x)^2 + (b.y - c.y)^2 = sideLength^2 ∧
    (c.x - a.x)^2 + (c.y - a.y)^2 = sideLength^2

-- Define an isosceles right triangle
structure IsoscelesRightTriangle where
  a : Point
  b : Point
  c : Point
  legLength : ℝ
  isIsoscelesRight :
    (a.x - b.x)^2 + (a.y - b.y)^2 = 2 * legLength^2 ∧
    (b.x - c.x)^2 + (b.y - c.y)^2 = legLength^2 ∧
    (c.x - a.x)^2 + (c.y - a.y)^2 = legLength^2

-- State the theorem
theorem colored_plane_triangles (coloring : Coloring) :
  (∃ t : EquilateralTriangle, 
    (t.sideLength = 673 * Real.sqrt 3 ∨ t.sideLength = 2019) ∧
    coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c) ∧
  (∃ t : IsoscelesRightTriangle,
    (t.legLength = 1010 * Real.sqrt 2 ∨ t.legLength = 2020) ∧
    coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c) :=
by sorry

end NUMINAMATH_CALUDE_colored_plane_triangles_l4027_402743


namespace NUMINAMATH_CALUDE_partnership_profit_theorem_l4027_402731

/-- Represents an investment in a partnership business -/
structure Investment where
  amount : ℕ
  duration : ℕ

/-- Calculates the total profit of a partnership business -/
def calculateTotalProfit (investments : List Investment) (cProfit : ℕ) : ℕ :=
  let totalCapitalMonths := investments.foldl (fun acc inv => acc + inv.amount * inv.duration) 0
  let cCapitalMonths := (investments.find? (fun inv => inv.amount = 6000 ∧ inv.duration = 6)).map (fun inv => inv.amount * inv.duration)
  match cCapitalMonths with
  | some cm => totalCapitalMonths * cProfit / cm
  | none => 0

theorem partnership_profit_theorem (investments : List Investment) (cProfit : ℕ) :
  investments = [
    ⟨8000, 12⟩,  -- A's investment
    ⟨4000, 8⟩,   -- B's investment
    ⟨6000, 6⟩,   -- C's investment
    ⟨10000, 9⟩   -- D's investment
  ] ∧ cProfit = 36000 →
  calculateTotalProfit investments cProfit = 254000 := by
  sorry

#eval calculateTotalProfit [⟨8000, 12⟩, ⟨4000, 8⟩, ⟨6000, 6⟩, ⟨10000, 9⟩] 36000

end NUMINAMATH_CALUDE_partnership_profit_theorem_l4027_402731


namespace NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_65_l4027_402735

theorem right_triangle_with_hypotenuse_65 :
  ∃ (a b : ℕ), 
    a < b ∧ 
    a^2 + b^2 = 65^2 ∧ 
    a = 16 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_65_l4027_402735


namespace NUMINAMATH_CALUDE_value_of_y_l4027_402760

theorem value_of_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : y = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l4027_402760


namespace NUMINAMATH_CALUDE_salary_expenditure_percentage_l4027_402705

theorem salary_expenditure_percentage (initial_salary : ℝ) 
  (house_rent_percentage : ℝ) (education_percentage : ℝ) 
  (final_amount : ℝ) : 
  initial_salary = 2125 →
  house_rent_percentage = 20 →
  education_percentage = 10 →
  final_amount = 1377 →
  let remaining_after_rent := initial_salary * (1 - house_rent_percentage / 100)
  let remaining_after_education := remaining_after_rent * (1 - education_percentage / 100)
  let clothes_percentage := (remaining_after_education - final_amount) / remaining_after_education * 100
  clothes_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_salary_expenditure_percentage_l4027_402705


namespace NUMINAMATH_CALUDE_min_value_theorem_l4027_402769

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ 1/x + 1/y = 1 → 1/(a-1) + 9/(b-1) ≤ 1/(x-1) + 9/(y-1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4027_402769


namespace NUMINAMATH_CALUDE_cube_root_problem_l4027_402718

theorem cube_root_problem (a : ℕ) (h : a^3 = 21 * 49 * 45 * 25) : a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l4027_402718


namespace NUMINAMATH_CALUDE_marbles_probability_l4027_402739

theorem marbles_probability (total : ℕ) (red : ℕ) (h1 : total = 48) (h2 : red = 12) :
  let p := (total - red) / total
  p * p = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_marbles_probability_l4027_402739


namespace NUMINAMATH_CALUDE_min_square_value_l4027_402722

theorem min_square_value (a b : ℕ+) 
  (h1 : ∃ m : ℕ+, (15 * a + 16 * b : ℕ) = m^2)
  (h2 : ∃ n : ℕ+, (16 * a - 15 * b : ℕ) = n^2) :
  min (15 * a + 16 * b) (16 * a - 15 * b) ≥ 481 :=
sorry

end NUMINAMATH_CALUDE_min_square_value_l4027_402722


namespace NUMINAMATH_CALUDE_circle_areas_and_square_l4027_402779

/-- Given two concentric circles with radii 23 and 33 units, prove that a third circle
    with area equal to the shaded area between the two original circles has a radius of 4√35,
    and when inscribed in a square, the square's side length is 8√35. -/
theorem circle_areas_and_square (r₁ r₂ r₃ : ℝ) (s : ℝ) : 
  r₁ = 23 →
  r₂ = 33 →
  π * r₃^2 = π * (r₂^2 - r₁^2) →
  s = 2 * r₃ →
  r₃ = 4 * Real.sqrt 35 ∧ s = 8 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_circle_areas_and_square_l4027_402779


namespace NUMINAMATH_CALUDE_expression_evaluation_l4027_402796

/-- Given a = -2 and b = -1/2, prove that the expression 3(2a²-4ab)-[a²-3(4a+ab)] evaluates to -13 -/
theorem expression_evaluation (a b : ℚ) (h1 : a = -2) (h2 : b = -1/2) :
  3 * (2 * a^2 - 4 * a * b) - (a^2 - 3 * (4 * a + a * b)) = -13 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4027_402796


namespace NUMINAMATH_CALUDE_M_remainder_mod_45_l4027_402758

def M : ℕ := sorry

theorem M_remainder_mod_45 : M % 45 = 15 := by sorry

end NUMINAMATH_CALUDE_M_remainder_mod_45_l4027_402758


namespace NUMINAMATH_CALUDE_square_max_perimeter_l4027_402748

/-- A right-angled quadrilateral inscribed in a circle --/
structure InscribedRightQuadrilateral (r : ℝ) where
  x : ℝ
  y : ℝ
  right_angled : x^2 + y^2 = (2*r)^2
  inscribed : x > 0 ∧ y > 0

/-- The perimeter of an inscribed right-angled quadrilateral --/
def perimeter (r : ℝ) (q : InscribedRightQuadrilateral r) : ℝ :=
  2 * (q.x + q.y)

/-- The statement that the square has the largest perimeter --/
theorem square_max_perimeter (r : ℝ) (hr : r > 0) :
  ∀ q : InscribedRightQuadrilateral r,
    perimeter r q ≤ 4 * r * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_square_max_perimeter_l4027_402748


namespace NUMINAMATH_CALUDE_a_value_is_two_l4027_402761

/-- Represents the chemical reaction 3A + B ⇌ aC + 2D -/
structure Reaction where
  a : ℕ

/-- Represents the reaction conditions -/
structure ReactionConditions where
  initial_A : ℝ
  initial_B : ℝ
  volume : ℝ
  time : ℝ
  final_C : ℝ
  rate_D : ℝ

/-- Determines the value of 'a' in the reaction equation -/
def determine_a (reaction : Reaction) (conditions : ReactionConditions) : ℕ :=
  sorry

/-- Theorem stating that the value of 'a' is 2 given the specified conditions -/
theorem a_value_is_two :
  ∀ (reaction : Reaction) (conditions : ReactionConditions),
    conditions.initial_A = 0.6 ∧
    conditions.initial_B = 0.5 ∧
    conditions.volume = 0.4 ∧
    conditions.time = 5 ∧
    conditions.final_C = 0.2 ∧
    conditions.rate_D = 0.1 →
    determine_a reaction conditions = 2 :=
  sorry

end NUMINAMATH_CALUDE_a_value_is_two_l4027_402761


namespace NUMINAMATH_CALUDE_dice_prime_probability_l4027_402794

def probability_prime : ℚ := 5 / 12

def number_of_dice : ℕ := 5

def target_prime_count : ℕ := 3

theorem dice_prime_probability :
  let p := probability_prime
  let n := number_of_dice
  let k := target_prime_count
  (n.choose k) * p^k * (1 - p)^(n - k) = 6125 / 24883 := by sorry

end NUMINAMATH_CALUDE_dice_prime_probability_l4027_402794


namespace NUMINAMATH_CALUDE_kennedy_gas_consumption_l4027_402713

-- Define the problem parameters
def miles_per_gallon : ℝ := 19
def distance_to_school : ℝ := 15
def distance_to_softball : ℝ := 6
def distance_to_restaurant : ℝ := 2
def distance_to_friend : ℝ := 4
def distance_to_home : ℝ := 11

-- Define the theorem
theorem kennedy_gas_consumption :
  let total_distance := distance_to_school + distance_to_softball + distance_to_restaurant + distance_to_friend + distance_to_home
  total_distance / miles_per_gallon = 2 := by
  sorry

end NUMINAMATH_CALUDE_kennedy_gas_consumption_l4027_402713


namespace NUMINAMATH_CALUDE_baseball_audience_percentage_l4027_402773

theorem baseball_audience_percentage (total : ℕ) (second_team_percent : ℚ) (neutral : ℕ) :
  total = 50 →
  second_team_percent = 34 / 100 →
  neutral = 3 →
  (total - (total * second_team_percent).floor - neutral) / total = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_baseball_audience_percentage_l4027_402773


namespace NUMINAMATH_CALUDE_initial_birds_count_l4027_402751

theorem initial_birds_count (initial_storks : ℕ) (additional_birds : ℕ) (total_after : ℕ) :
  initial_storks = 2 →
  additional_birds = 5 →
  total_after = 10 →
  ∃ initial_birds : ℕ, initial_birds + initial_storks + additional_birds = total_after ∧ initial_birds = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_birds_count_l4027_402751


namespace NUMINAMATH_CALUDE_m_range_l4027_402723

theorem m_range : ∃ m : ℝ, m = 3 * Real.sqrt 2 - 1 ∧ 3 < m ∧ m < 4 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l4027_402723


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l4027_402726

/-- Proves that 448000 is equal to 4.48 * 10^5 in scientific notation -/
theorem scientific_notation_equality : 448000 = 4.48 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l4027_402726


namespace NUMINAMATH_CALUDE_investment_result_l4027_402753

/-- Calculates the future value of an investment with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that an investment of $4000 at 10% annual compound interest for 2 years results in $4840 -/
theorem investment_result : compound_interest 4000 0.1 2 = 4840 := by
  sorry

end NUMINAMATH_CALUDE_investment_result_l4027_402753


namespace NUMINAMATH_CALUDE_largest_possible_reflections_l4027_402725

/-- Represents the angle of reflection at each point -/
def reflection_angle (n : ℕ) : ℝ := 15 * n

/-- The condition for the beam to hit perpendicularly and retrace its path -/
def valid_reflection (n : ℕ) : Prop := reflection_angle n ≤ 90

theorem largest_possible_reflections : ∃ (max_n : ℕ), 
  (∀ n : ℕ, valid_reflection n → n ≤ max_n) ∧ 
  valid_reflection max_n ∧ 
  max_n = 6 :=
sorry

end NUMINAMATH_CALUDE_largest_possible_reflections_l4027_402725


namespace NUMINAMATH_CALUDE_unique_divisible_by_13_l4027_402730

def base_7_to_10 (d : Nat) : Nat :=
  3 * 7^3 + d * 7^2 + d * 7 + 6

theorem unique_divisible_by_13 : 
  ∃! d : Nat, d < 7 ∧ (base_7_to_10 d) % 13 = 0 ∧ base_7_to_10 d = 1035 + 56 * d :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_13_l4027_402730


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4027_402777

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x - 7
  ∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 11 ∧ x2 = 2 - Real.sqrt 11 ∧ f x1 = 0 ∧ f x2 = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x1 ∨ x = x2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4027_402777


namespace NUMINAMATH_CALUDE_sequence_fourth_term_l4027_402797

theorem sequence_fourth_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n^3) : a 4 = 37 := by
  sorry

end NUMINAMATH_CALUDE_sequence_fourth_term_l4027_402797


namespace NUMINAMATH_CALUDE_bubble_gum_cost_l4027_402798

theorem bubble_gum_cost (total_cost : ℕ) (total_pieces : ℕ) (cost_per_piece : ℕ) : 
  total_cost = 2448 → 
  total_pieces = 136 → 
  total_cost = total_pieces * cost_per_piece → 
  cost_per_piece = 18 := by
  sorry

end NUMINAMATH_CALUDE_bubble_gum_cost_l4027_402798


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_cube_l4027_402787

/-- The volume of a sphere that circumscribes a cube with side length 2 is 4√3π. -/
theorem sphere_volume_circumscribing_cube (cube_side : ℝ) (sphere_volume : ℝ) : 
  cube_side = 2 →
  sphere_volume = (4 / 3) * Real.pi * (Real.sqrt 3 * cube_side / 2)^3 →
  sphere_volume = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_circumscribing_cube_l4027_402787


namespace NUMINAMATH_CALUDE_basketball_preference_theorem_l4027_402736

/-- Represents the school population and basketball preferences -/
structure School where
  total_students : ℕ
  male_ratio : ℚ
  female_ratio : ℚ
  male_basketball_ratio : ℚ
  female_basketball_ratio : ℚ

/-- Calculate the percentage of students who do not like basketball -/
def percentage_not_liking_basketball (s : School) : ℚ :=
  let male_students := s.total_students * s.male_ratio / (s.male_ratio + s.female_ratio)
  let female_students := s.total_students * s.female_ratio / (s.male_ratio + s.female_ratio)
  let male_liking_basketball := male_students * s.male_basketball_ratio
  let female_liking_basketball := female_students * s.female_basketball_ratio
  let total_not_liking := s.total_students - (male_liking_basketball + female_liking_basketball)
  total_not_liking / s.total_students * 100

/-- The main theorem to prove -/
theorem basketball_preference_theorem (s : School) 
  (h1 : s.total_students = 1000)
  (h2 : s.male_ratio = 3)
  (h3 : s.female_ratio = 2)
  (h4 : s.male_basketball_ratio = 2/3)
  (h5 : s.female_basketball_ratio = 1/5) :
  percentage_not_liking_basketball s = 52 := by
  sorry


end NUMINAMATH_CALUDE_basketball_preference_theorem_l4027_402736


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l4027_402770

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  length : ℕ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.width + r.length)

/-- Theorem: A rectangle with perimeter 150 cm and length 15 cm greater than width
    has width 30 cm and length 45 cm -/
theorem rectangle_dimensions :
  ∃ (r : Rectangle),
    perimeter r = 150 ∧
    r.length = r.width + 15 ∧
    r.width = 30 ∧
    r.length = 45 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l4027_402770


namespace NUMINAMATH_CALUDE_cube_difference_l4027_402749

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 50) :
  a^3 - b^3 = 353.5 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l4027_402749


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l4027_402790

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 8 * X^3 + 16 * X^2 - 7 * X + 4
  let divisor : Polynomial ℚ := 2 * X + 5
  let quotient : Polynomial ℚ := 4 * X^2 - 2 * X + (3/2)
  dividend = divisor * quotient + (-7/2) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l4027_402790


namespace NUMINAMATH_CALUDE_surface_area_increase_l4027_402781

/-- Given a cube with edge length a that is cut into 27 identical smaller cubes,
    the increase in surface area is 12a². -/
theorem surface_area_increase (a : ℝ) (h : a > 0) : 
  27 * 6 * (a / 3)^2 - 6 * a^2 = 12 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_increase_l4027_402781


namespace NUMINAMATH_CALUDE_vector_magnitude_l4027_402768

/-- Given two vectors a and b in a 2D space, if the angle between them is 120°,
    |a| = 2, and |a + b| = √7, then |b| = 3. -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  let θ := Real.arccos (-1/2)  -- 120° in radians
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt ((a.1^2 + a.2^2) * (b.1^2 + b.2^2))) = Real.cos θ →
  Real.sqrt (a.1^2 + a.2^2) = 2 →
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 7 →
  Real.sqrt (b.1^2 + b.2^2) = 3 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_l4027_402768


namespace NUMINAMATH_CALUDE_parabola_transformation_theorem_l4027_402703

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Rotates a parabola 180 degrees about its vertex -/
def rotate180 (p : Parabola) : Parabola :=
  { a := -p.a, h := p.h, k := p.k }

/-- Shifts a parabola horizontally -/
def shiftHorizontal (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, h := p.h - shift, k := p.k }

/-- Shifts a parabola vertically -/
def shiftVertical (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, h := p.h, k := p.k + shift }

/-- Calculates the sum of zeros for a parabola -/
def sumOfZeros (p : Parabola) : ℝ := 2 * p.h

theorem parabola_transformation_theorem :
  let original := Parabola.mk 1 3 4
  let transformed := shiftVertical (shiftHorizontal (rotate180 original) 5) (-4)
  sumOfZeros transformed = 16 := by sorry

end NUMINAMATH_CALUDE_parabola_transformation_theorem_l4027_402703


namespace NUMINAMATH_CALUDE_cookie_bags_count_l4027_402744

theorem cookie_bags_count (cookies_per_bag : ℕ) (total_cookies : ℕ) (h1 : cookies_per_bag = 19) (h2 : total_cookies = 703) :
  total_cookies / cookies_per_bag = 37 := by
  sorry

end NUMINAMATH_CALUDE_cookie_bags_count_l4027_402744


namespace NUMINAMATH_CALUDE_salary_problem_l4027_402782

theorem salary_problem (total : ℝ) (a_spend_rate : ℝ) (b_spend_rate : ℝ) 
  (h1 : total = 6000)
  (h2 : a_spend_rate = 0.95)
  (h3 : b_spend_rate = 0.85)
  (h4 : (1 - a_spend_rate) * a = (1 - b_spend_rate) * (total - a)) :
  a = 4500 :=
by
  sorry

end NUMINAMATH_CALUDE_salary_problem_l4027_402782


namespace NUMINAMATH_CALUDE_joey_fraction_of_ethan_time_l4027_402721

def alexa_vacation_days : ℕ := 7 + 2  -- 1 week and 2 days

def joey_learning_days : ℕ := 6

def alexa_vacation_fraction : ℚ := 3/4

theorem joey_fraction_of_ethan_time : 
  (joey_learning_days : ℚ) / ((alexa_vacation_days : ℚ) / alexa_vacation_fraction) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_joey_fraction_of_ethan_time_l4027_402721


namespace NUMINAMATH_CALUDE_distribute_6_balls_3_boxes_l4027_402780

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distributeIndistinguishable (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 5 ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem distribute_6_balls_3_boxes : distributeIndistinguishable 6 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_6_balls_3_boxes_l4027_402780


namespace NUMINAMATH_CALUDE_sphere_triangle_distance_is_four_l4027_402745

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Represents a triangle with given side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Calculates the distance between the center of a sphere and the plane of a triangle tangent to it -/
def sphereTriangleDistance (s : Sphere) (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating the distance between the sphere's center and the triangle's plane -/
theorem sphere_triangle_distance_is_four :
  ∀ (s : Sphere) (t : Triangle),
    s.radius = 8 ∧
    t.side1 = 13 ∧ t.side2 = 14 ∧ t.side3 = 15 →
    sphereTriangleDistance s t = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_triangle_distance_is_four_l4027_402745


namespace NUMINAMATH_CALUDE_correct_balloons_left_l4027_402752

/-- Given the number of balloons of each color and the number of friends,
    calculate the number of balloons left after even distribution. -/
def balloons_left (yellow blue pink violet friends : ℕ) : ℕ :=
  let total := yellow + blue + pink + violet
  total % friends

theorem correct_balloons_left :
  balloons_left 20 24 50 102 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_correct_balloons_left_l4027_402752


namespace NUMINAMATH_CALUDE_cosine_set_product_l4027_402702

open Real Set

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

def S (a₁ : ℝ) : Set ℝ := {x | ∃ n : ℕ+, x = cos (arithmeticSequence a₁ (2 * π / 3) n)}

theorem cosine_set_product (a₁ : ℝ) :
  (∃ a b : ℝ, S a₁ = {a, b} ∧ a ≠ b) → 
  ∀ a b : ℝ, S a₁ = {a, b} → a * b = -1/2 := by
sorry

end NUMINAMATH_CALUDE_cosine_set_product_l4027_402702


namespace NUMINAMATH_CALUDE_polynomial_identity_l4027_402786

theorem polynomial_identity (P : ℝ → ℝ) : 
  (∀ x, P x - 3 * x = 5 * x^2 - 3 * x - 5) → 
  (∀ x, P x = 5 * x^2 - 5) := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l4027_402786


namespace NUMINAMATH_CALUDE_inequality_transformations_l4027_402776

theorem inequality_transformations :
  (∀ x : ℝ, x - 1 > 2 → x > 3) ∧
  (∀ x : ℝ, -4 * x > 8 → x < -2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformations_l4027_402776


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l4027_402756

/-- For a parabola with equation y² = 8x, the distance from its focus to its directrix is 4. -/
theorem parabola_focus_directrix_distance (x y : ℝ) : 
  y^2 = 8*x → (distance_focus_to_directrix : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l4027_402756


namespace NUMINAMATH_CALUDE_no_divisible_by_ten_l4027_402784

/-- The function g(x) = x^2 + 5x + 3 -/
def g (x : ℤ) : ℤ := x^2 + 5*x + 3

/-- The set T of integers from 0 to 30 -/
def T : Set ℤ := {x | 0 ≤ x ∧ x ≤ 30}

/-- Theorem: There are no integers t in T such that g(t) is divisible by 10 -/
theorem no_divisible_by_ten : ∀ t ∈ T, ¬(g t % 10 = 0) := by sorry

end NUMINAMATH_CALUDE_no_divisible_by_ten_l4027_402784


namespace NUMINAMATH_CALUDE_frustum_radius_l4027_402712

theorem frustum_radius (r : ℝ) 
  (h1 : (2 * π * (3 * r)) / (2 * π * r) = 3)
  (h2 : 3 = 3)  -- slant height
  (h3 : π * (r + 3 * r) * 3 = 84 * π) : r = 7 := by
  sorry

end NUMINAMATH_CALUDE_frustum_radius_l4027_402712


namespace NUMINAMATH_CALUDE_min_value_sum_squares_and_reciprocals_l4027_402788

theorem min_value_sum_squares_and_reciprocals (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 + b^2 + 1/a^2 + 1/b^2 ≥ 4 ∧
  (a^2 + b^2 + 1/a^2 + 1/b^2 = 4 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_and_reciprocals_l4027_402788


namespace NUMINAMATH_CALUDE_horner_method_v1_l4027_402772

def f (x : ℝ) : ℝ := 3 * x^4 + 2 * x^2 + x + 4

def horner_v1 (a : ℝ) : ℝ := 3 * a + 0

theorem horner_method_v1 :
  let x : ℝ := 10
  horner_v1 x = 30 := by sorry

end NUMINAMATH_CALUDE_horner_method_v1_l4027_402772


namespace NUMINAMATH_CALUDE_range_of_a_l4027_402778

-- Define the * operation
def star (x y : ℝ) := x * (1 - y)

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, star x (x - a) > 0 → -1 ≤ x ∧ x ≤ 1) → 
  -2 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4027_402778


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l4027_402755

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 17*x + 8 = 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 17*x + 8 = (x - r₁) * (x - r₂) ∧ 
  (1 / r₁ + 1 / r₂ = 17 / 8) := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l4027_402755


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l4027_402742

/-- The amount of material Cheryl used for her project -/
def material_used (material1 material2 leftover : ℚ) : ℚ :=
  material1 + material2 - leftover

/-- Theorem stating the total amount of material Cheryl used -/
theorem cheryl_material_usage :
  let material1 : ℚ := 5 / 11
  let material2 : ℚ := 2 / 3
  let leftover : ℚ := 25 / 55
  material_used material1 material2 leftover = 22 / 33 := by
sorry

#eval material_used (5/11) (2/3) (25/55)

end NUMINAMATH_CALUDE_cheryl_material_usage_l4027_402742


namespace NUMINAMATH_CALUDE_f_properties_l4027_402714

-- Define the function f
def f (x b c : ℝ) : ℝ := x * |x| + b * x + c

-- Theorem statement
theorem f_properties :
  (∀ x, f x 0 0 = -f (-x) 0 0) ∧
  (∀ x, f x 0 (0 : ℝ) = 0 → x = 0) ∧
  (∀ x, f (x - 0) b c = f (-x - 0) b c + 2 * c) ∧
  (∃ b c, ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x b c = 0 ∧ f y b c = 0 ∧ f z b c = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4027_402714


namespace NUMINAMATH_CALUDE_power_function_inequality_l4027_402799

theorem power_function_inequality : let f : ℝ → ℝ := fun x ↦ x^3
  let a : ℝ := f (Real.sqrt 3 / 3)
  let b : ℝ := f (Real.log π)
  let c : ℝ := f (Real.sqrt 2 / 2)
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_power_function_inequality_l4027_402799


namespace NUMINAMATH_CALUDE_f_increasing_iff_three_distinct_roots_iff_l4027_402708

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * |2 * a - x| + 2 * x

-- Theorem 1: f(x) is increasing on ℝ iff -1 ≤ a ≤ 1
theorem f_increasing_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ -1 ≤ a ∧ a ≤ 1 :=
sorry

-- Theorem 2: f(x) - t f(2a) = 0 has 3 distinct real roots iff 1 < t < 9/8
theorem three_distinct_roots_iff (a t : ℝ) :
  (a ∈ Set.Icc (-2) 2) →
  (∃ x y z : ℝ, x < y ∧ y < z ∧ f a x = t * f a (2 * a) ∧ f a y = t * f a (2 * a) ∧ f a z = t * f a (2 * a)) ↔
  (1 < t ∧ t < 9/8) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_three_distinct_roots_iff_l4027_402708


namespace NUMINAMATH_CALUDE_min_valid_configuration_l4027_402711

/-- Represents a configuration of two piles of bricks -/
structure BrickPiles where
  first : ℕ
  second : ℕ

/-- Checks if moving 100 bricks from the first pile to the second makes the second pile twice as large as the first -/
def satisfiesFirstCondition (piles : BrickPiles) : Prop :=
  2 * (piles.first - 100) = piles.second + 100

/-- Checks if there exists a number of bricks that can be moved from the second pile to the first to make the first pile six times as large as the second -/
def satisfiesSecondCondition (piles : BrickPiles) : Prop :=
  ∃ z : ℕ, piles.first + z = 6 * (piles.second - z)

/-- Checks if a given configuration satisfies both conditions -/
def isValidConfiguration (piles : BrickPiles) : Prop :=
  satisfiesFirstCondition piles ∧ satisfiesSecondCondition piles

/-- The main theorem stating the minimum valid configuration -/
theorem min_valid_configuration :
  ∀ piles : BrickPiles, isValidConfiguration piles →
  piles.first ≥ 170 ∧
  (piles.first = 170 → piles.second = 40) :=
by sorry

#check min_valid_configuration

end NUMINAMATH_CALUDE_min_valid_configuration_l4027_402711


namespace NUMINAMATH_CALUDE_sin_x_squared_not_periodic_l4027_402791

theorem sin_x_squared_not_periodic : ¬ ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, Real.sin ((x + p)^2) = Real.sin (x^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_x_squared_not_periodic_l4027_402791


namespace NUMINAMATH_CALUDE_inequality_solution_l4027_402715

theorem inequality_solution (a : ℝ) (h : |a + 1| < 3) :
  (∀ x, x - (a + 1) * (x + 1) > 0 ↔ 
    ((-4 < a ∧ a < -2 ∧ (x > -1 ∨ x < 1 + a)) ∨
     (a = -2 ∧ x ≠ -1) ∨
     (-2 < a ∧ a < 2 ∧ (x > 1 + a ∨ x < -1)))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4027_402715


namespace NUMINAMATH_CALUDE_pet_shop_inventory_l4027_402706

/-- Given a pet shop with dogs, cats, and bunnies in stock, prove the total number of dogs and bunnies. -/
theorem pet_shop_inventory (dogs cats bunnies : ℕ) : 
  dogs = 112 →
  dogs / bunnies = 4 / 9 →
  dogs / cats = 4 / 7 →
  dogs + bunnies = 364 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_inventory_l4027_402706


namespace NUMINAMATH_CALUDE_abc_value_l4027_402719

theorem abc_value (a b c : ℝ) 
  (sum_condition : a + b + c = 1)
  (sum_squares : a^2 + b^2 + c^2 = 2)
  (sum_cubes : a^3 + b^3 + c^3 = 3) :
  a * b * c = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_abc_value_l4027_402719


namespace NUMINAMATH_CALUDE_participation_plans_specific_l4027_402724

/-- The number of ways to select three students from four, with one student always selected,
    for three different subjects. -/
def participation_plans (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  (n - 1).choose (k - 1) * m.factorial

theorem participation_plans_specific : participation_plans 4 3 3 = 18 := by
  sorry

#eval participation_plans 4 3 3

end NUMINAMATH_CALUDE_participation_plans_specific_l4027_402724


namespace NUMINAMATH_CALUDE_game_not_fair_l4027_402720

/-- Represents the game described in the problem -/
structure Game where
  deck_size : ℕ
  named_cards : ℕ
  win_amount : ℚ
  lose_amount : ℚ

/-- Calculates the expected winnings for the guessing player -/
def expected_winnings (g : Game) : ℚ :=
  let p_named := g.named_cards / g.deck_size
  let p_not_named := 1 - p_named
  let max_cards_per_suit := g.deck_size / 4
  let p_correct_guess_not_named := max_cards_per_suit / (g.deck_size - g.named_cards)
  let expected_case1 := p_named * g.win_amount
  let expected_case2 := p_not_named * (p_correct_guess_not_named * g.win_amount - (1 - p_correct_guess_not_named) * g.lose_amount)
  expected_case1 + expected_case2

/-- The theorem stating that the expected winnings for the guessing player are 1/8 Ft -/
theorem game_not_fair (g : Game) (h1 : g.deck_size = 32) (h2 : g.named_cards = 4) 
    (h3 : g.win_amount = 2) (h4 : g.lose_amount = 1) : 
  expected_winnings g = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_game_not_fair_l4027_402720


namespace NUMINAMATH_CALUDE_solve_for_x_l4027_402727

theorem solve_for_x (x : ℝ) : 
  let M := 2*x - 2
  let N := 2*x + 3
  2*M - N = 1 → x = 4 := by
sorry

end NUMINAMATH_CALUDE_solve_for_x_l4027_402727


namespace NUMINAMATH_CALUDE_triangle_perimeter_l4027_402795

theorem triangle_perimeter (a b c : ℕ) : 
  a = 2 → b = 4 → Even c → 
  a + b > c ∧ b + c > a ∧ c + a > b → 
  a + b + c = 10 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l4027_402795


namespace NUMINAMATH_CALUDE_max_cards_no_sum_l4027_402750

/-- Given a positive integer k, prove that from 2k+1 cards numbered 1 to 2k+1,
    the maximum number of cards that can be selected such that no selected number
    is the sum of two other selected numbers is k+1. -/
theorem max_cards_no_sum (k : ℕ) : ∃ (S : Finset ℕ),
  S.card = k + 1 ∧
  S.toSet ⊆ Finset.range (2*k + 2) ∧
  (∀ x ∈ S, ∀ y ∈ S, ∀ z ∈ S, x + y ≠ z) ∧
  (∀ T : Finset ℕ, T.toSet ⊆ Finset.range (2*k + 2) →
    (∀ x ∈ T, ∀ y ∈ T, ∀ z ∈ T, x + y ≠ z) →
    T.card ≤ k + 1) :=
sorry

end NUMINAMATH_CALUDE_max_cards_no_sum_l4027_402750


namespace NUMINAMATH_CALUDE_y_derivative_l4027_402729

noncomputable def y (x : ℝ) : ℝ :=
  (2 * x^2 - x + 1/2) * Real.arctan ((x^2 - 1) / (x * Real.sqrt 3)) - 
  x^3 / (2 * Real.sqrt 3) - (Real.sqrt 3 / 2) * x

theorem y_derivative (x : ℝ) (hx : x ≠ 0) : 
  deriv y x = (4 * x - 1) * Real.arctan ((x^2 - 1) / (x * Real.sqrt 3)) + 
  (Real.sqrt 3 * (x^2 + 1) * (3 * x^2 - 2 * x - x^4)) / (2 * (x^4 + x^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l4027_402729


namespace NUMINAMATH_CALUDE_f_derivatives_l4027_402701

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x - 1

-- Theorem statement
theorem f_derivatives :
  (deriv f 2 = 0) ∧ (deriv f 1 = -1) := by
  sorry

end NUMINAMATH_CALUDE_f_derivatives_l4027_402701


namespace NUMINAMATH_CALUDE_complex_inside_unit_circle_l4027_402709

theorem complex_inside_unit_circle (x : ℝ) :
  (∀ z : ℂ, z = x - (1/3 : ℝ) * Complex.I → Complex.abs z < 1) →
  -2 * Real.sqrt 2 / 3 < x ∧ x < 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_inside_unit_circle_l4027_402709


namespace NUMINAMATH_CALUDE_f_properties_l4027_402741

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - 2*a| + |x - a|

theorem f_properties (a : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, (a = 1 ∧ f 1 x > 3) ↔ (x < 0 ∨ x > 3)) ∧
  (∀ b : ℝ, b ≠ 0 → f a b ≥ f a a) ∧
  (∀ b : ℝ, b ≠ 0 → (f a b = f a a ↔ (2*a - b) * (b - a) ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4027_402741


namespace NUMINAMATH_CALUDE_valentines_day_theorem_l4027_402765

theorem valentines_day_theorem (male_students female_students : ℕ) : 
  (male_students * female_students = male_students + female_students + 42) → 
  (male_students * female_students = 88) :=
by
  sorry

end NUMINAMATH_CALUDE_valentines_day_theorem_l4027_402765


namespace NUMINAMATH_CALUDE_line_slope_equals_y_coord_l4027_402762

/-- Given a line passing through points (-1, -4) and (4, y), 
    if the slope of the line is equal to y, then y = 1. -/
theorem line_slope_equals_y_coord (y : ℝ) : 
  (y - (-4)) / (4 - (-1)) = y → y = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_equals_y_coord_l4027_402762
