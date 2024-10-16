import Mathlib

namespace NUMINAMATH_CALUDE_fishing_competition_l3111_311118

/-- Fishing competition problem -/
theorem fishing_competition 
  (days : ℕ) 
  (jackson_per_day : ℕ) 
  (jonah_per_day : ℕ) 
  (total_catch : ℕ) :
  days = 5 →
  jackson_per_day = 6 →
  jonah_per_day = 4 →
  total_catch = 90 →
  ∃ (george_per_day : ℕ), 
    george_per_day = 8 ∧ 
    days * (jackson_per_day + jonah_per_day + george_per_day) = total_catch :=
by sorry

end NUMINAMATH_CALUDE_fishing_competition_l3111_311118


namespace NUMINAMATH_CALUDE_digit_appearance_l3111_311140

def digit_free (n : ℕ) (d : Finset ℕ) : Prop :=
  ∀ (i : ℕ), i ∈ d → (n / 10^i % 10 ≠ i)

def contains_digit (n : ℕ) (d : Finset ℕ) : Prop :=
  ∃ (i : ℕ), i ∈ d ∧ (n / 10^i % 10 = i)

theorem digit_appearance (n : ℕ) (h1 : n ≥ 1) (h2 : digit_free n {1, 2, 9}) :
  contains_digit (3 * n) {1, 2, 9} := by
  sorry

end NUMINAMATH_CALUDE_digit_appearance_l3111_311140


namespace NUMINAMATH_CALUDE_tangent_slope_is_e_l3111_311196

/-- The exponential function -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x

/-- A line passing through the origin -/
def line_through_origin (k : ℝ) (x : ℝ) : ℝ := k * x

/-- Tangent condition: The line touches the curve at exactly one point -/
def is_tangent (k : ℝ) : Prop :=
  ∃ x₀ : ℝ, 
    f x₀ = line_through_origin k x₀ ∧
    ∀ x ≠ x₀, f x ≠ line_through_origin k x

theorem tangent_slope_is_e :
  ∃ k : ℝ, is_tangent k ∧ k = Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_is_e_l3111_311196


namespace NUMINAMATH_CALUDE_steve_union_dues_l3111_311104

/-- Calculate the amount lost to local union dues given gross salary, tax rate, healthcare rate, and take-home pay -/
def union_dues (gross_salary : ℝ) (tax_rate : ℝ) (healthcare_rate : ℝ) (take_home_pay : ℝ) : ℝ :=
  gross_salary - (tax_rate * gross_salary) - (healthcare_rate * gross_salary) - take_home_pay

/-- Theorem: Given Steve's financial information, prove that he loses $800 to local union dues -/
theorem steve_union_dues :
  union_dues 40000 0.20 0.10 27200 = 800 := by
  sorry

end NUMINAMATH_CALUDE_steve_union_dues_l3111_311104


namespace NUMINAMATH_CALUDE_min_sum_with_linear_constraint_l3111_311124

theorem min_sum_with_linear_constraint (a b : ℕ) (h : 23 * a - 13 * b = 1) :
  ∃ (a' b' : ℕ), 23 * a' - 13 * b' = 1 ∧ a' + b' ≤ a + b ∧ a' + b' = 11 :=
sorry

end NUMINAMATH_CALUDE_min_sum_with_linear_constraint_l3111_311124


namespace NUMINAMATH_CALUDE_measles_cases_1995_l3111_311110

/-- Represents the number of measles cases in a given year -/
def measles_cases (year : ℕ) : ℝ :=
  if year ≤ 1990 then
    300000 - 14950 * (year - 1970)
  else
    -8 * (year - 1990)^2 + 1000

/-- The theorem stating that the number of measles cases in 1995 is 800 -/
theorem measles_cases_1995 : measles_cases 1995 = 800 := by
  sorry

end NUMINAMATH_CALUDE_measles_cases_1995_l3111_311110


namespace NUMINAMATH_CALUDE_mary_nickels_l3111_311101

theorem mary_nickels (initial_nickels : ℕ) (dad_gave_nickels : ℕ) 
  (h1 : initial_nickels = 7)
  (h2 : dad_gave_nickels = 5) : 
  initial_nickels + dad_gave_nickels = 12 := by
sorry

end NUMINAMATH_CALUDE_mary_nickels_l3111_311101


namespace NUMINAMATH_CALUDE_income_left_is_2_15_percent_l3111_311180

/-- Calculates the percentage of income left after one year given initial expenses and yearly changes. -/
def income_left_after_one_year (
  food_expense : ℝ)
  (education_expense : ℝ)
  (transportation_expense : ℝ)
  (medical_expense : ℝ)
  (rent_percentage_of_remaining : ℝ)
  (expense_increase_rate : ℝ)
  (income_increase_rate : ℝ) : ℝ :=
  let initial_expenses := food_expense + education_expense + transportation_expense + medical_expense
  let remaining_after_initial := 1 - initial_expenses
  let initial_rent := remaining_after_initial * rent_percentage_of_remaining
  let increased_expenses := initial_expenses * (1 + expense_increase_rate)
  let new_remaining := 1 - increased_expenses
  let new_rent := new_remaining * rent_percentage_of_remaining
  1 - (increased_expenses + new_rent)

/-- Theorem stating that given the specified conditions, the percentage of income left after one year is 2.15%. -/
theorem income_left_is_2_15_percent :
  income_left_after_one_year 0.35 0.25 0.15 0.10 0.80 0.05 0.10 = 0.0215 := by
  sorry

end NUMINAMATH_CALUDE_income_left_is_2_15_percent_l3111_311180


namespace NUMINAMATH_CALUDE_sequence_periodicity_l3111_311150

def is_periodic (a : ℕ → ℤ) : Prop :=
  ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, a (n + p) = a n

theorem sequence_periodicity (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, n ≥ 2 → (0 : ℝ) ≤ a (n - 1) + ((1 - Real.sqrt 5) / 2) * (a n) + a (n + 1) ∧
                       a (n - 1) + ((1 - Real.sqrt 5) / 2) * (a n) + a (n + 1) < 1) :
  is_periodic a :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l3111_311150


namespace NUMINAMATH_CALUDE_managers_salary_l3111_311184

/-- Proves that the manager's salary is 14100 given the conditions -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℕ) (salary_increase : ℕ) :
  num_employees = 20 →
  avg_salary = 1500 →
  salary_increase = 600 →
  (num_employees * avg_salary + (num_employees + 1) * salary_increase) = 14100 := by
  sorry

#check managers_salary

end NUMINAMATH_CALUDE_managers_salary_l3111_311184


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l3111_311137

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B⁻¹ = !![3, -2; 1, 1] →
  (B^3)⁻¹ = !![13, -22; 11, -9] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l3111_311137


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l3111_311119

theorem unique_solution_to_equation (x : ℝ) : (x^2 + 4*x - 5)^0 = x^2 - 5*x + 5 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l3111_311119


namespace NUMINAMATH_CALUDE_largest_positive_integer_satisfying_condition_l3111_311185

theorem largest_positive_integer_satisfying_condition : 
  ∀ x : ℕ+, x + 1000 > 1000 * x.val → x.val ≤ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_positive_integer_satisfying_condition_l3111_311185


namespace NUMINAMATH_CALUDE_stars_per_bottle_l3111_311159

/-- Given Shiela's paper stars and number of classmates, prove the number of stars per bottle. -/
theorem stars_per_bottle (total_stars : ℕ) (num_classmates : ℕ) 
  (h1 : total_stars = 45) 
  (h2 : num_classmates = 9) : 
  total_stars / num_classmates = 5 := by
  sorry

#check stars_per_bottle

end NUMINAMATH_CALUDE_stars_per_bottle_l3111_311159


namespace NUMINAMATH_CALUDE_minimizing_integral_minimizing_function_achieves_minimum_minimizing_function_integral_one_l3111_311123

noncomputable def minimizing_function (x : ℝ) : ℝ := 6 / (Real.pi * (x^2 + x + 1))

theorem minimizing_integral 
  (f : ℝ → ℝ) 
  (hf_continuous : Continuous f) 
  (hf_integral : ∫ x in (0:ℝ)..1, f x = 1) :
  ∫ x in (0:ℝ)..1, (x^2 + x + 1) * (f x)^2 ≥ 6 / Real.pi :=
sorry

theorem minimizing_function_achieves_minimum :
  ∫ x in (0:ℝ)..1, (x^2 + x + 1) * (minimizing_function x)^2 = 6 / Real.pi :=
sorry

theorem minimizing_function_integral_one :
  ∫ x in (0:ℝ)..1, minimizing_function x = 1 :=
sorry

end NUMINAMATH_CALUDE_minimizing_integral_minimizing_function_achieves_minimum_minimizing_function_integral_one_l3111_311123


namespace NUMINAMATH_CALUDE_four_digit_perfect_square_palindrome_l3111_311113

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem four_digit_perfect_square_palindrome :
  ∃! n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n := by sorry

end NUMINAMATH_CALUDE_four_digit_perfect_square_palindrome_l3111_311113


namespace NUMINAMATH_CALUDE_log_product_equality_l3111_311166

theorem log_product_equality : (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) *
  (Real.log 2 / Real.log 3 + Real.log 8 / Real.log 9) = 25 / 12 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equality_l3111_311166


namespace NUMINAMATH_CALUDE_store_owner_order_theorem_l3111_311156

/-- The number of bottles of soda ordered by a store owner in April and May -/
def total_bottles_ordered (april_cases : ℕ) (may_cases : ℕ) (bottles_per_case : ℕ) : ℕ :=
  (april_cases + may_cases) * bottles_per_case

/-- Theorem stating that the store owner ordered 1000 bottles in April and May -/
theorem store_owner_order_theorem :
  total_bottles_ordered 20 30 20 = 1000 := by
  sorry

#eval total_bottles_ordered 20 30 20

end NUMINAMATH_CALUDE_store_owner_order_theorem_l3111_311156


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3111_311191

theorem condition_sufficient_not_necessary : 
  (∃ (S T : Set ℝ), 
    (S = {x : ℝ | x - 1 > 0}) ∧ 
    (T = {x : ℝ | x^2 - 1 > 0}) ∧ 
    (S ⊂ T) ∧ 
    (∃ x, x ∈ T ∧ x ∉ S)) := by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3111_311191


namespace NUMINAMATH_CALUDE_horner_method_V₂_l3111_311122

-- Define the polynomial coefficients
def a₄ : ℤ := 3
def a₃ : ℤ := 5
def a₂ : ℤ := 6
def a₁ : ℤ := 79
def a₀ : ℤ := -8

-- Define the x value
def x : ℤ := -4

-- Define Horner's method steps
def V₀ : ℤ := a₄
def V₁ : ℤ := x * V₀ + a₃
def V₂ : ℤ := x * V₁ + a₂

-- Theorem statement
theorem horner_method_V₂ : V₂ = 34 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_V₂_l3111_311122


namespace NUMINAMATH_CALUDE_circle_radius_problem_l3111_311162

theorem circle_radius_problem (r : ℝ) : 
  3 * (2 * Real.pi * r) + 6 = 2 * (Real.pi * r^2) → 
  r = (3 + Real.sqrt 21) / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l3111_311162


namespace NUMINAMATH_CALUDE_ellipse_properties_l3111_311109

-- Define the ellipse G
def ellipse (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / 4 = 1

-- Define the foci
def left_focus (c : ℝ) : ℝ × ℝ := (-c, 0)
def right_focus (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define a point on the ellipse
def point_on_ellipse (a : ℝ) (M : ℝ × ℝ) : Prop := ellipse a M.1 M.2

-- Define perpendicularity condition
def perpendicular (M F₁ F₂ : ℝ × ℝ) : Prop :=
  (M.1 - F₂.1) * (F₂.1 - F₁.1) + (M.2 - F₂.2) * (F₂.2 - F₁.2) = 0

-- Define the distance difference condition
def distance_diff (M F₁ F₂ : ℝ × ℝ) (a : ℝ) : Prop :=
  Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) -
  Real.sqrt ((M.2 - F₂.1)^2 + (M.2 - F₂.2)^2) = 4*a/3

-- Define the theorem
theorem ellipse_properties (a c : ℝ) (M : ℝ × ℝ) 
  (h_a_pos : a > 0)
  (h_on_ellipse : point_on_ellipse a M)
  (h_perp : perpendicular M (left_focus c) (right_focus c))
  (h_dist : distance_diff M (left_focus c) (right_focus c) a) :
  (∀ x y, ellipse a x y ↔ x^2 / 12 + y^2 / 4 = 1) ∧
  (∃ A B : ℝ × ℝ, 
    ellipse a A.1 A.2 ∧ 
    ellipse a B.1 B.2 ∧
    B.2 - A.2 = B.1 - A.1 ∧ 
    (let P : ℝ × ℝ := (-3, 2);
     let S := (B.1 - A.1) * (P.2 - A.2) - (B.2 - A.2) * (P.1 - A.1);
     S * S / 2 = 9/2)) := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3111_311109


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_18_l3111_311158

theorem consecutive_even_numbers_sum_18 (n : ℤ) : 
  (n - 2) + n + (n + 2) = 18 → (n - 2 = 4 ∧ n = 6 ∧ n + 2 = 8) := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_18_l3111_311158


namespace NUMINAMATH_CALUDE_smallest_n_for_probability_threshold_l3111_311177

def P (n : ℕ) : ℚ := 3 / ((n + 1) * (n + 2) * (n + 3))

theorem smallest_n_for_probability_threshold : 
  ∀ k : ℕ, k ≥ 1 → (P k < 1 / 3015 ↔ k ≥ 19) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_probability_threshold_l3111_311177


namespace NUMINAMATH_CALUDE_coordinate_transform_sum_l3111_311139

/-- Definition of the original coordinate system -/
structure OriginalCoord where
  x : ℝ
  y : ℝ

/-- Definition of the new coordinate system -/
structure NewCoord where
  x : ℝ
  y : ℝ

/-- Definition of a line -/
structure Line where
  slope : ℝ
  point : OriginalCoord

/-- Function to transform coordinates from original to new system -/
def transform (p : OriginalCoord) (L M : Line) : NewCoord :=
  sorry

/-- Theorem statement -/
theorem coordinate_transform_sum :
  let A : OriginalCoord := ⟨24, -1⟩
  let B : OriginalCoord := ⟨5, 6⟩
  let P : OriginalCoord := ⟨-14, 27⟩
  let L : Line := ⟨5/12, A⟩
  let M : Line := ⟨-12/5, B⟩  -- Perpendicular slope
  let new_P : NewCoord := transform P L M
  new_P.x + new_P.y = 31 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_transform_sum_l3111_311139


namespace NUMINAMATH_CALUDE_p_range_l3111_311157

/-- The function p(x) defined for x ≥ 0 -/
def p (x : ℝ) : ℝ := x^4 + 8*x^2 + 16

/-- The range of p(x) is [16, ∞) -/
theorem p_range :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ p x = y) ↔ y ≥ 16 := by sorry

end NUMINAMATH_CALUDE_p_range_l3111_311157


namespace NUMINAMATH_CALUDE_probability_same_fruit_choices_l3111_311198

/-- The number of fruit types available -/
def num_fruits : ℕ := 4

/-- The number of fruit types each student must choose -/
def num_choices : ℕ := 2

/-- The probability that two students choose the same two types of fruits -/
def probability_same_choice : ℚ := 1 / 6

/-- Theorem stating the probability of two students choosing the same fruits -/
theorem probability_same_fruit_choices :
  (Nat.choose num_fruits num_choices : ℚ) / ((Nat.choose num_fruits num_choices : ℚ) ^ 2) = probability_same_choice :=
sorry

end NUMINAMATH_CALUDE_probability_same_fruit_choices_l3111_311198


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l3111_311163

/-- Proof that when m(m-1) + mi is purely imaginary, m = 1 -/
theorem purely_imaginary_condition (m : ℝ) : 
  (m * (m - 1) : ℂ) + m * Complex.I = Complex.I * (r : ℝ) → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l3111_311163


namespace NUMINAMATH_CALUDE_book_arrangement_count_l3111_311129

/-- The number of ways to arrange books on a shelf --/
def arrange_books (n_science : ℕ) (n_literature : ℕ) : ℕ :=
  n_literature * (n_literature - 1) * Nat.factorial (n_science + n_literature - 2)

/-- Theorem stating the number of arrangements for the given problem --/
theorem book_arrangement_count :
  arrange_books 4 5 = 10080 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l3111_311129


namespace NUMINAMATH_CALUDE_sector_area_l3111_311121

theorem sector_area (θ : Real) (arc_length : Real) (area : Real) : 
  θ = π / 3 →  -- 60° in radians
  arc_length = 2 * π → 
  area = 6 * π :=
by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3111_311121


namespace NUMINAMATH_CALUDE_product_remainder_mod_nine_l3111_311108

theorem product_remainder_mod_nine : (2156 * 4427 * 9313) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_nine_l3111_311108


namespace NUMINAMATH_CALUDE_count_power_functions_l3111_311116

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = x^k

def f₁ (x : ℝ) : ℝ := x^3
def f₂ (x : ℝ) : ℝ := 4*x^2
def f₃ (x : ℝ) : ℝ := x^5 + 1
def f₄ (x : ℝ) : ℝ := (x-1)^2
def f₅ (x : ℝ) : ℝ := x

theorem count_power_functions : 
  (is_power_function f₁ ∧ ¬is_power_function f₂ ∧ ¬is_power_function f₃ ∧ 
   ¬is_power_function f₄ ∧ is_power_function f₅) :=
by sorry

end NUMINAMATH_CALUDE_count_power_functions_l3111_311116


namespace NUMINAMATH_CALUDE_base8_to_base10_conversion_l3111_311100

/-- Converts a base 8 number to base 10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base 8 representation of the number --/
def base8Number : List Nat := [3, 4, 2]

theorem base8_to_base10_conversion :
  base8ToBase10 base8Number = 163 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base10_conversion_l3111_311100


namespace NUMINAMATH_CALUDE_fish_tank_capacity_l3111_311105

/-- The capacity of a fish tank given pouring rate, duration, and remaining volume --/
theorem fish_tank_capacity
  (pour_rate : ℚ)  -- Pouring rate in gallons per second
  (pour_duration : ℕ)  -- Pouring duration in minutes
  (remaining_volume : ℕ)  -- Remaining volume to fill the tank in gallons
  (h1 : pour_rate = 1 / 20)  -- 1 gallon every 20 seconds
  (h2 : pour_duration = 6)  -- Poured for 6 minutes
  (h3 : remaining_volume = 32)  -- 32 more gallons needed
  : ℕ :=
by
  sorry

#check fish_tank_capacity

end NUMINAMATH_CALUDE_fish_tank_capacity_l3111_311105


namespace NUMINAMATH_CALUDE_captain_age_proof_l3111_311103

def cricket_team_problem (team_size : ℕ) (team_avg_age : ℕ) (age_diff : ℕ) (remaining_avg_diff : ℕ) : Prop :=
  let captain_age : ℕ := 26
  let keeper_age : ℕ := captain_age + age_diff
  let total_age : ℕ := team_size * team_avg_age
  let remaining_players : ℕ := team_size - 2
  let remaining_avg : ℕ := team_avg_age - remaining_avg_diff
  total_age = captain_age + keeper_age + remaining_players * remaining_avg

theorem captain_age_proof :
  cricket_team_problem 11 23 3 1 := by
  sorry

end NUMINAMATH_CALUDE_captain_age_proof_l3111_311103


namespace NUMINAMATH_CALUDE_shortest_tangent_theorem_l3111_311194

noncomputable def circle_C3 (x y : ℝ) : Prop := (x - 8) ^ 2 + (y - 3) ^ 2 = 49

noncomputable def circle_C4 (x y : ℝ) : Prop := (x + 12) ^ 2 + (y + 4) ^ 2 = 16

noncomputable def shortest_tangent_length : ℝ := (Real.sqrt 7840 + Real.sqrt 24181) / 11 - 11

theorem shortest_tangent_theorem :
  ∃ (R S : ℝ × ℝ),
    circle_C3 R.1 R.2 ∧
    circle_C4 S.1 S.2 ∧
    (∀ (P Q : ℝ × ℝ),
      circle_C3 P.1 P.2 →
      circle_C4 Q.1 Q.2 →
      Real.sqrt ((R.1 - S.1) ^ 2 + (R.2 - S.2) ^ 2) ≤ Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)) ∧
    Real.sqrt ((R.1 - S.1) ^ 2 + (R.2 - S.2) ^ 2) = shortest_tangent_length :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_tangent_theorem_l3111_311194


namespace NUMINAMATH_CALUDE_skittles_distribution_l3111_311135

theorem skittles_distribution (num_friends : ℕ) (total_skittles : ℕ) 
  (h1 : num_friends = 5) 
  (h2 : total_skittles = 200) : 
  total_skittles / num_friends = 40 := by
  sorry

end NUMINAMATH_CALUDE_skittles_distribution_l3111_311135


namespace NUMINAMATH_CALUDE_min_PM_AB_implies_line_AB_l3111_311151

/-- Circle M with equation x^2 + y^2 - 2x - 2y - 2 = 0 -/
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y - 2 = 0

/-- Line l with equation 2x + y + 2 = 0 -/
def line_l (x y : ℝ) : Prop :=
  2*x + y + 2 = 0

/-- Point P on line l -/
structure Point_P where
  x : ℝ
  y : ℝ
  on_line_l : line_l x y

/-- Tangent line from P to circle M -/
def is_tangent (P : Point_P) (A : ℝ × ℝ) : Prop :=
  circle_M A.1 A.2 ∧ 
  ∃ (t : ℝ), A.1 = P.x + t * (A.1 - P.x) ∧ A.2 = P.y + t * (A.2 - P.y)

/-- The equation of line AB: 2x + y + 1 = 0 -/
def line_AB (x y : ℝ) : Prop :=
  2*x + y + 1 = 0

theorem min_PM_AB_implies_line_AB :
  ∀ (P : Point_P) (A B : ℝ × ℝ),
  is_tangent P A → is_tangent P B →
  (∀ (Q : Point_P) (C D : ℝ × ℝ),
    is_tangent Q C → is_tangent Q D →
    (P.x - 1)^2 + (P.y - 1)^2 * ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤
    (Q.x - 1)^2 + (Q.y - 1)^2 * ((C.1 - D.1)^2 + (C.2 - D.2)^2)) →
  line_AB A.1 A.2 ∧ line_AB B.1 B.2 := by
  sorry

end NUMINAMATH_CALUDE_min_PM_AB_implies_line_AB_l3111_311151


namespace NUMINAMATH_CALUDE_prime_square_mod_180_l3111_311192

theorem prime_square_mod_180 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_5 : p > 5) :
  ∃ (s : Finset ℕ), (∀ x ∈ s, x < 180) ∧ (Finset.card s = 2) ∧
  (∀ q : ℕ, Nat.Prime q → q > 5 → (q^2 % 180) ∈ s) :=
sorry

end NUMINAMATH_CALUDE_prime_square_mod_180_l3111_311192


namespace NUMINAMATH_CALUDE_fourth_row_from_bottom_sum_l3111_311127

/-- Represents a position in the grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents the grid and its properties -/
structure Grid :=
  (size : ℕ)
  (start : Position)
  (max_num : ℕ)

/-- Represents the spiral filling of the grid -/
def spiral_fill (g : Grid) : Position → ℕ := sorry

/-- The sum of the greatest and least number in a given row -/
def row_sum (g : Grid) (row : ℕ) : ℕ := sorry

/-- The main theorem to prove -/
theorem fourth_row_from_bottom_sum :
  let g : Grid := {
    size := 16,
    start := { row := 8, col := 8 },
    max_num := 256
  }
  row_sum g 4 = 497 := by sorry

end NUMINAMATH_CALUDE_fourth_row_from_bottom_sum_l3111_311127


namespace NUMINAMATH_CALUDE_min_absolute_value_complex_l3111_311128

theorem min_absolute_value_complex (z : ℂ) (h : Complex.abs (z - 10) + Complex.abs (z + 3*I) = 15) :
  ∃ (w : ℂ), Complex.abs w = 2 ∧ ∀ (v : ℂ), (Complex.abs (v - 10) + Complex.abs (v + 3*I) = 15) → Complex.abs v ≥ Complex.abs w :=
by sorry

end NUMINAMATH_CALUDE_min_absolute_value_complex_l3111_311128


namespace NUMINAMATH_CALUDE_four_square_product_l3111_311197

theorem four_square_product (p q r s p₁ q₁ r₁ s₁ : ℝ) :
  ∃ A B C D : ℝ, (p^2 + q^2 + r^2 + s^2) * (p₁^2 + q₁^2 + r₁^2 + s₁^2) = A^2 + B^2 + C^2 + D^2 := by
  sorry

end NUMINAMATH_CALUDE_four_square_product_l3111_311197


namespace NUMINAMATH_CALUDE_product_of_divisors_2022_no_prime_power_2022_l3111_311186

def T (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem product_of_divisors_2022 : T 2022 = 2022^4 := by sorry

theorem no_prime_power_2022 : ∀ (n : ℕ) (p : ℕ), Nat.Prime p → T n ≠ p^2022 := by sorry

end NUMINAMATH_CALUDE_product_of_divisors_2022_no_prime_power_2022_l3111_311186


namespace NUMINAMATH_CALUDE_probability_ratio_l3111_311143

def total_slips : ℕ := 40
def distinct_numbers : ℕ := 8
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 4

def probability_same_number (n : ℕ) : ℚ :=
  (n * slips_per_number.choose drawn_slips) / total_slips.choose drawn_slips

def probability_two_pairs (n : ℕ) : ℚ :=
  (n.choose 2 * slips_per_number.choose 2 * slips_per_number.choose 2) / total_slips.choose drawn_slips

theorem probability_ratio :
  probability_two_pairs distinct_numbers / probability_same_number distinct_numbers = 70 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l3111_311143


namespace NUMINAMATH_CALUDE_smallest_result_l3111_311175

def S : Set ℕ := {6, 8, 10, 12, 14, 16}

def process (a b c : ℕ) : ℕ := (a + b) * c - 10

def valid_choice (a b c : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_result :
  ∀ a b c : ℕ, valid_choice a b c →
    98 ≤ min (process a b c) (min (process a c b) (process b c a)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_result_l3111_311175


namespace NUMINAMATH_CALUDE_green_ball_probability_l3111_311155

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers in the problem -/
def containers : List Container := [
  ⟨10, 2⟩,  -- Container I
  ⟨3, 5⟩,   -- Container II
  ⟨2, 6⟩,   -- Container III
  ⟨5, 3⟩    -- Container IV
]

/-- The probability of selecting each container -/
def containerProbability : ℚ := 1 / containers.length

theorem green_ball_probability : 
  (containers.map (fun c => containerProbability * greenProbability c)).sum = 23 / 48 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l3111_311155


namespace NUMINAMATH_CALUDE_max_a_value_l3111_311182

theorem max_a_value (x y a : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 17) 
  (h4 : (3/4) * x = (5/6) * y + a) (h5 : a > 0) : a < 51/4 := by
  sorry

end NUMINAMATH_CALUDE_max_a_value_l3111_311182


namespace NUMINAMATH_CALUDE_parabola_equation_correct_l3111_311106

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola -/
structure Parabola where
  focus : Point
  directrix : Line

/-- The equation of a parabola in general form -/
def parabola_equation (p : Parabola) (x y : ℝ) : Prop :=
  x^2 - 2*x*y + y^2 - 12*x - 16*y + 78 = 0

theorem parabola_equation_correct (p : Parabola) :
  p.focus = Point.mk 4 5 →
  p.directrix = Line.mk 1 1 (-2) →
  ∀ x y : ℝ, (x^2 - 2*x*y + y^2 - 12*x - 16*y + 78 = 0) ↔
    (((x - 4)^2 + (y - 5)^2) = ((x + y - 2)^2 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_correct_l3111_311106


namespace NUMINAMATH_CALUDE_rowing_coach_votes_l3111_311183

theorem rowing_coach_votes (num_coaches : ℕ) (votes_per_rower : ℕ) (votes_per_coach : ℕ) :
  num_coaches = 50 →
  votes_per_rower = 4 →
  votes_per_coach = 7 →
  ∃ (num_rowers : ℕ), num_rowers * votes_per_rower = num_coaches * votes_per_coach ∧ 
                       num_rowers = 88 := by
  sorry

end NUMINAMATH_CALUDE_rowing_coach_votes_l3111_311183


namespace NUMINAMATH_CALUDE_translation_result_l3111_311173

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translates a point in the x-direction -/
def translateX (p : Point2D) (dx : ℝ) : Point2D :=
  { x := p.x + dx, y := p.y }

/-- Translates a point in the y-direction -/
def translateY (p : Point2D) (dy : ℝ) : Point2D :=
  { x := p.x, y := p.y + dy }

/-- The initial point P -/
def P : Point2D := { x := -2, y := 3 }

theorem translation_result :
  let p1 := translateX P 3
  let p2 := translateY p1 (-2)
  p2 = { x := 1, y := 1 } := by sorry

end NUMINAMATH_CALUDE_translation_result_l3111_311173


namespace NUMINAMATH_CALUDE_books_read_per_year_l3111_311131

/-- The number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  36 * c * s

/-- Theorem stating the total number of books read by the entire student body in one year -/
theorem books_read_per_year (c s : ℕ) : 
  total_books_read c s = 3 * 12 * c * s := by
  sorry

#check books_read_per_year

end NUMINAMATH_CALUDE_books_read_per_year_l3111_311131


namespace NUMINAMATH_CALUDE_hill_height_l3111_311149

/-- The height of a hill given its base depth and proportion to total vertical distance -/
theorem hill_height (base_depth : ℝ) (total_distance : ℝ) 
  (h1 : base_depth = 300)
  (h2 : base_depth = (1/4) * total_distance) : 
  total_distance - base_depth = 900 := by
  sorry

end NUMINAMATH_CALUDE_hill_height_l3111_311149


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3111_311168

theorem min_value_of_expression (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let A := (a^2 + b^2)^4 / (c*d)^4 + (b^2 + c^2)^4 / (a*d)^4 + (c^2 + d^2)^4 / (a*b)^4 + (d^2 + a^2)^4 / (b*c)^4
  A ≥ 64 ∧ (A = 64 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3111_311168


namespace NUMINAMATH_CALUDE_f_monotonicity_and_equality_l3111_311153

noncomputable def f (x : ℝ) : ℝ := (Real.exp 1) * x / Real.exp x

theorem f_monotonicity_and_equality (e : ℝ) (he : e = Real.exp 1) :
  (∀ x y, x < y → x < 1 → y < 1 → f x < f y) ∧
  (∀ x y, x < y → 1 < x → 1 < y → f y < f x) ∧
  (∀ x, x > 0 → f (1 - x) ≠ f (1 + x)) ∧
  (f (1 - 0) = f (1 + 0)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_equality_l3111_311153


namespace NUMINAMATH_CALUDE_balloon_distribution_l3111_311164

theorem balloon_distribution (yellow_balloons : ℕ) (black_balloon_difference : ℕ) (num_schools : ℕ) : 
  yellow_balloons = 3414 →
  black_balloon_difference = 1762 →
  num_schools = 10 →
  (yellow_balloons + (yellow_balloons + black_balloon_difference)) / num_schools = 859 :=
by sorry

end NUMINAMATH_CALUDE_balloon_distribution_l3111_311164


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l3111_311172

theorem least_number_with_remainder (n : ℕ) : n = 282 ↔ 
  (n > 0 ∧ 
   n % 31 = 3 ∧ 
   n % 9 = 3 ∧ 
   ∀ m : ℕ, m > 0 → m % 31 = 3 → m % 9 = 3 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l3111_311172


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l3111_311142

/-- Represents the number of ways to arrange frogs with color restrictions -/
def frog_arrangements (n_green n_red n_blue : ℕ) : ℕ :=
  2 * (n_red.factorial * n_green.factorial)

/-- Theorem stating the number of valid frog arrangements -/
theorem frog_arrangement_count :
  frog_arrangements 3 4 1 = 288 :=
by sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l3111_311142


namespace NUMINAMATH_CALUDE_max_type_B_bins_l3111_311171

def unit_price_A : ℕ := 300
def unit_price_B : ℕ := 450
def total_budget : ℕ := 8000
def total_bins : ℕ := 20

theorem max_type_B_bins :
  ∀ y : ℕ,
    y ≤ 13 ∧
    y ≤ total_bins ∧
    unit_price_A * (total_bins - y) + unit_price_B * y ≤ total_budget ∧
    (∀ z : ℕ, z > y →
      z > 13 ∨
      z > total_bins ∨
      unit_price_A * (total_bins - z) + unit_price_B * z > total_budget) :=
by sorry

end NUMINAMATH_CALUDE_max_type_B_bins_l3111_311171


namespace NUMINAMATH_CALUDE_salaria_trees_count_l3111_311111

/-- Represents the total number of trees Salaria has -/
def total_trees : ℕ := sorry

/-- Represents the number of oranges tree A produces per month -/
def tree_A_oranges : ℕ := 10

/-- Represents the number of oranges tree B produces per month -/
def tree_B_oranges : ℕ := 15

/-- Represents the fraction of good oranges from tree A -/
def tree_A_good_fraction : ℚ := 3/5

/-- Represents the fraction of good oranges from tree B -/
def tree_B_good_fraction : ℚ := 1/3

/-- Represents the total number of good oranges Salaria gets per month -/
def total_good_oranges : ℕ := 55

theorem salaria_trees_count :
  total_trees = 10 ∧
  (total_trees / 2 : ℚ) * tree_A_oranges * tree_A_good_fraction +
  (total_trees / 2 : ℚ) * tree_B_oranges * tree_B_good_fraction = total_good_oranges := by
  sorry

end NUMINAMATH_CALUDE_salaria_trees_count_l3111_311111


namespace NUMINAMATH_CALUDE_tangent_two_implications_l3111_311174

open Real

theorem tangent_two_implications (α : ℝ) (h : tan α = 2) :
  (sin α + 2 * cos α) / (4 * cos α - sin α) = 2 ∧
  Real.sqrt 2 * sin (2 * α + π / 4) + 1 = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_two_implications_l3111_311174


namespace NUMINAMATH_CALUDE_certain_number_problem_l3111_311133

theorem certain_number_problem : 
  ∃ x : ℝ, (3500 - (x / 20.50) = 3451.2195121951218) ∧ (x = 1000) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3111_311133


namespace NUMINAMATH_CALUDE_isosceles_triangle_triangle_area_l3111_311120

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem isosceles_triangle (t : Triangle) (h : t.a * Real.sin t.A = t.b * Real.sin t.B) :
  t.a = t.b := by
  sorry

-- Part 2
theorem triangle_area (t : Triangle) 
  (h1 : t.a + t.b = t.a * t.b)
  (h2 : t.c = 2)
  (h3 : t.C = π / 3) :
  (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_triangle_area_l3111_311120


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l3111_311167

/-- Given a man's speed with the current and the speed of the current,
    calculates the man's speed against the current. -/
def speedAgainstCurrent (speedWithCurrent : ℝ) (currentSpeed : ℝ) : ℝ :=
  speedWithCurrent - 2 * currentSpeed

/-- Theorem stating that given the specific conditions of the problem,
    the man's speed against the current is 18 kmph. -/
theorem mans_speed_against_current :
  speedAgainstCurrent 20 1 = 18 := by
  sorry

#eval speedAgainstCurrent 20 1

end NUMINAMATH_CALUDE_mans_speed_against_current_l3111_311167


namespace NUMINAMATH_CALUDE_pizza_toppings_l3111_311165

theorem pizza_toppings (total_slices : ℕ) (cheese_slices : ℕ) (pepperoni_slices : ℕ)
  (h1 : total_slices = 16)
  (h2 : cheese_slices = 10)
  (h3 : pepperoni_slices = 12)
  (h4 : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range cheese_slices ∨ slice ∈ Finset.range pepperoni_slices)) :
  cheese_slices + pepperoni_slices - total_slices = 6 :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l3111_311165


namespace NUMINAMATH_CALUDE_roof_area_is_400_l3111_311199

/-- Represents a rectangular roof with given properties -/
structure RectangularRoof where
  width : ℝ
  length : ℝ
  length_is_triple_width : length = 3 * width
  length_width_difference : length - width = 30

/-- The area of a rectangular roof -/
def roof_area (roof : RectangularRoof) : ℝ :=
  roof.length * roof.width

/-- Theorem stating that a roof with the given properties has an area of 400 square feet -/
theorem roof_area_is_400 (roof : RectangularRoof) : roof_area roof = 400 := by
  sorry

end NUMINAMATH_CALUDE_roof_area_is_400_l3111_311199


namespace NUMINAMATH_CALUDE_median_is_twelve_l3111_311148

def group_sizes : List ℕ := [10, 10, 8]

def median (l : List ℕ) (x : ℕ) : ℚ :=
  sorry

theorem median_is_twelve (x : ℕ) : median (x :: group_sizes) x = 12 :=
  sorry

end NUMINAMATH_CALUDE_median_is_twelve_l3111_311148


namespace NUMINAMATH_CALUDE_supermarket_spending_l3111_311136

theorem supermarket_spending (total : ℚ) : 
  (1/2 : ℚ) * total + (1/3 : ℚ) * total + (1/10 : ℚ) * total + 10 = total →
  total = 150 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l3111_311136


namespace NUMINAMATH_CALUDE_average_salary_non_officers_l3111_311102

/-- Proof of the average salary of non-officers in an office --/
theorem average_salary_non_officers
  (total_avg : ℝ)
  (officer_avg : ℝ)
  (officer_count : ℕ)
  (non_officer_count : ℕ)
  (h_total_avg : total_avg = 120)
  (h_officer_avg : officer_avg = 430)
  (h_officer_count : officer_count = 15)
  (h_non_officer_count : non_officer_count = 465) :
  let non_officer_avg := (((officer_count + non_officer_count) * total_avg) - (officer_count * officer_avg)) / non_officer_count
  non_officer_avg = 110 := by
sorry


end NUMINAMATH_CALUDE_average_salary_non_officers_l3111_311102


namespace NUMINAMATH_CALUDE_desk_height_in_cm_mm_l3111_311126

/-- The height of a chair in millimeters -/
def chair_height : ℕ := 537

/-- Dong-min's height when standing on the chair, in millimeters -/
def height_on_chair : ℕ := 1900

/-- Dong-min's height when standing on the desk, in millimeters -/
def height_on_desk : ℕ := 2325

/-- The height of the desk in millimeters -/
def desk_height : ℕ := height_on_desk - (height_on_chair - chair_height)

theorem desk_height_in_cm_mm : 
  desk_height = 96 * 10 + 2 := by sorry

end NUMINAMATH_CALUDE_desk_height_in_cm_mm_l3111_311126


namespace NUMINAMATH_CALUDE_remainder_seven_205_mod_12_l3111_311114

theorem remainder_seven_205_mod_12 : 7^205 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_205_mod_12_l3111_311114


namespace NUMINAMATH_CALUDE_lineup_ways_eq_choose_four_from_fourteen_l3111_311154

/-- The number of ways to choose an 8-player lineup from 18 players,
    including two sets of twins that must be in the lineup. -/
def lineup_ways (total_players : ℕ) (lineup_size : ℕ) (twin_pairs : ℕ) : ℕ :=
  Nat.choose (total_players - 2 * twin_pairs) (lineup_size - 2 * twin_pairs)

/-- Theorem stating that the number of ways to choose the lineup
    is equal to choosing 4 from 14 players. -/
theorem lineup_ways_eq_choose_four_from_fourteen :
  lineup_ways 18 8 2 = Nat.choose 14 4 := by
  sorry

end NUMINAMATH_CALUDE_lineup_ways_eq_choose_four_from_fourteen_l3111_311154


namespace NUMINAMATH_CALUDE_product_quality_probability_l3111_311179

theorem product_quality_probability (p_B p_C : ℝ) 
  (h_B : p_B = 0.03) 
  (h_C : p_C = 0.02) : 
  1 - (p_B + p_C) = 0.95 := by
  sorry

end NUMINAMATH_CALUDE_product_quality_probability_l3111_311179


namespace NUMINAMATH_CALUDE_prove_income_expenditure_ratio_l3111_311181

def income_expenditure_ratio (income savings : ℕ) : Prop :=
  ∃ (expenditure : ℕ),
    savings = income - expenditure ∧
    income * 8 = expenditure * 15

theorem prove_income_expenditure_ratio :
  income_expenditure_ratio 15000 7000 := by
  sorry

end NUMINAMATH_CALUDE_prove_income_expenditure_ratio_l3111_311181


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3111_311193

theorem cubic_root_sum (α β γ : ℂ) : 
  (α^3 - 7*α^2 + 11*α - 13 = 0) →
  (β^3 - 7*β^2 + 11*β - 13 = 0) →
  (γ^3 - 7*γ^2 + 11*γ - 13 = 0) →
  (α*β/γ + β*γ/α + γ*α/β = -61/13) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3111_311193


namespace NUMINAMATH_CALUDE_ball_weight_order_l3111_311132

theorem ball_weight_order (a b c d : ℝ) 
  (eq1 : a + b = c + d)
  (ineq1 : a + d > b + c)
  (ineq2 : a + c < b) :
  d > b ∧ b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ball_weight_order_l3111_311132


namespace NUMINAMATH_CALUDE_muffin_to_banana_ratio_l3111_311187

/-- The cost of a muffin -/
def muffin_cost : ℝ := sorry

/-- The cost of a banana -/
def banana_cost : ℝ := sorry

/-- Kristy's total cost -/
def kristy_cost : ℝ := 5 * muffin_cost + 4 * banana_cost

/-- Tim's total cost -/
def tim_cost : ℝ := 3 * muffin_cost + 20 * banana_cost

/-- The theorem stating the ratio of muffin cost to banana cost -/
theorem muffin_to_banana_ratio :
  tim_cost = 3 * kristy_cost →
  muffin_cost = (2/3) * banana_cost :=
by sorry

end NUMINAMATH_CALUDE_muffin_to_banana_ratio_l3111_311187


namespace NUMINAMATH_CALUDE_imaginary_axis_length_of_given_hyperbola_l3111_311160

/-- The length of the imaginary axis of a hyperbola -/
def imaginary_axis_length (a b : ℝ) : ℝ := 2 * b

/-- The equation of the hyperbola in standard form -/
def hyperbola_equation (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

theorem imaginary_axis_length_of_given_hyperbola :
  ∃ (a b : ℝ), a^2 = 3 ∧ b^2 = 1 ∧
  (∀ x y : ℝ, hyperbola_equation x y a b ↔ x^2 / 3 - y^2 = 1) ∧
  imaginary_axis_length a b = 2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_axis_length_of_given_hyperbola_l3111_311160


namespace NUMINAMATH_CALUDE_function_inequality_l3111_311115

-- Define the functions f and g
def f (x b : ℝ) : ℝ := |x + b^2| - |-x + 1|
def g (x a b c : ℝ) : ℝ := |x + a^2 + c^2| + |x - 2*b^2|

-- State the theorem
theorem function_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a*b + b*c + a*c = 1) :
  (∀ x : ℝ, f x 1 ≥ 1 ↔ x ∈ Set.Ici (1/2)) ∧
  (∀ x : ℝ, f x b ≤ g x a b c) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3111_311115


namespace NUMINAMATH_CALUDE_set_relationship_l3111_311117

def M : Set ℤ := {x | ∃ k : ℤ, x = 5 * k - 2}
def P : Set ℤ := {x | ∃ n : ℤ, x = 5 * n + 3}
def S : Set ℤ := {x | ∃ m : ℤ, x = 10 * m + 3}

theorem set_relationship : S ⊆ P ∧ P = M := by sorry

end NUMINAMATH_CALUDE_set_relationship_l3111_311117


namespace NUMINAMATH_CALUDE_line_2x_plus_1_not_in_fourth_quadrant_l3111_311144

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Defines the fourth quadrant of the 2D plane -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Checks if a given line passes through the fourth quadrant -/
def passes_through_fourth_quadrant (l : Line) : Prop :=
  ∃ x y : ℝ, y = l.slope * x + l.y_intercept ∧ fourth_quadrant x y

/-- The main theorem stating that the line y = 2x + 1 does not pass through the fourth quadrant -/
theorem line_2x_plus_1_not_in_fourth_quadrant :
  ¬ passes_through_fourth_quadrant (Line.mk 2 1) := by
  sorry


end NUMINAMATH_CALUDE_line_2x_plus_1_not_in_fourth_quadrant_l3111_311144


namespace NUMINAMATH_CALUDE_euler_product_theorem_l3111_311176

theorem euler_product_theorem : ∀ (z₁ z₂ : ℂ),
  (z₁ = Complex.exp (Complex.I * Real.pi / 3)) →
  (z₂ = Complex.exp (Complex.I * Real.pi / 6)) →
  z₁ * z₂ = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_euler_product_theorem_l3111_311176


namespace NUMINAMATH_CALUDE_alex_shirts_l3111_311161

theorem alex_shirts (alex joe ben : ℕ) 
  (h1 : joe = alex + 3) 
  (h2 : ben = joe + 8) 
  (h3 : ben = 15) : 
  alex = 4 := by
sorry

end NUMINAMATH_CALUDE_alex_shirts_l3111_311161


namespace NUMINAMATH_CALUDE_triangle_inequality_l3111_311188

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (Real.sqrt (b + c - a) / (Real.sqrt b + Real.sqrt c - Real.sqrt a)) +
  (Real.sqrt (c + a - b) / (Real.sqrt c + Real.sqrt a - Real.sqrt b)) +
  (Real.sqrt (a + b - c) / (Real.sqrt a + Real.sqrt b - Real.sqrt c)) ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3111_311188


namespace NUMINAMATH_CALUDE_hanoi_moves_correct_l3111_311134

/-- The minimum number of moves required to solve the Tower of Hanoi problem with n disks -/
def hanoi_moves (n : ℕ) : ℕ := 2^n - 1

/-- Theorem: The minimum number of moves required to solve the Tower of Hanoi problem with n disks is 2^n - 1 -/
theorem hanoi_moves_correct (n : ℕ) : hanoi_moves n = 2^n - 1 := by
  sorry

end NUMINAMATH_CALUDE_hanoi_moves_correct_l3111_311134


namespace NUMINAMATH_CALUDE_identical_pairs_imply_x_equals_8_l3111_311130

-- Define the binary operation ★
def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

-- Theorem statement
theorem identical_pairs_imply_x_equals_8 :
  ∀ y : ℤ, star 5 7 1 3 = star x y 4 5 → x = 8 :=
by sorry

end NUMINAMATH_CALUDE_identical_pairs_imply_x_equals_8_l3111_311130


namespace NUMINAMATH_CALUDE_computer_price_increase_l3111_311107

theorem computer_price_increase (c : ℝ) (h : 2 * c = 540) : 
  (351 - c) / c * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l3111_311107


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3111_311147

theorem fraction_multiplication : (1/4 - 1/2 + 2/3) * (-12 : ℚ) = -8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3111_311147


namespace NUMINAMATH_CALUDE_mary_shirts_problem_l3111_311178

theorem mary_shirts_problem (blue_shirts : ℕ) (brown_shirts : ℕ) : 
  brown_shirts = 36 →
  blue_shirts / 2 + brown_shirts * 2 / 3 = 37 →
  blue_shirts = 26 := by
sorry

end NUMINAMATH_CALUDE_mary_shirts_problem_l3111_311178


namespace NUMINAMATH_CALUDE_cos_squared_alpha_plus_pi_fourth_l3111_311125

theorem cos_squared_alpha_plus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + π / 4) ^ 2 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_plus_pi_fourth_l3111_311125


namespace NUMINAMATH_CALUDE_solve_equation_l3111_311189

theorem solve_equation : 
  ∃ x : ℝ, (2 * x + 10 = (1/2) * (5 * x + 30)) ∧ (x = -10) := by sorry

end NUMINAMATH_CALUDE_solve_equation_l3111_311189


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l3111_311152

theorem divisibility_implies_equality (a b : ℕ+) (h : (a + b) ∣ (5 * a + 3 * b)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l3111_311152


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l3111_311170

theorem min_value_sum_fractions (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) :
  let u := (3*a^2 - a)/(1 + a^2) + (3*b^2 - b)/(1 + b^2) + (3*c^2 - c)/(1 + c^2)
  u ≥ 0 ∧ (u = 0 ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l3111_311170


namespace NUMINAMATH_CALUDE_system_two_solutions_l3111_311146

open Real

-- Define the system of equations
def equation1 (a x y : ℝ) : Prop :=
  arcsin ((a + y) / 2) = arcsin ((x + 3) / 3)

def equation2 (b x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x + 6*y = b

-- Define the condition for exactly two solutions
def hasTwoSolutions (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    equation1 a x₁ y₁ ∧ equation1 a x₂ y₂ ∧
    equation2 b x₁ y₁ ∧ equation2 b x₂ y₂ ∧
    ∀ (x y : ℝ), equation1 a x y ∧ equation2 b x y → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)

-- Theorem statement
theorem system_two_solutions (a : ℝ) :
  (∃ b, hasTwoSolutions a b) ↔ -7/2 < a ∧ a < 19/2 :=
sorry

end NUMINAMATH_CALUDE_system_two_solutions_l3111_311146


namespace NUMINAMATH_CALUDE_population_growth_theorem_l3111_311195

/-- The population growth rate from t=0 to t=1 -/
def growth_rate_0_to_1 : ℝ := 0.05

/-- The population growth rate from t=1 to t=2 -/
def growth_rate_1_to_2 : ℝ := 0.10

/-- The population growth rate from t=2 to t=3 -/
def growth_rate_2_to_3 : ℝ := 0.15

/-- The total growth factor from t=0 to t=3 -/
def total_growth_factor : ℝ := (1 + growth_rate_0_to_1) * (1 + growth_rate_1_to_2) * (1 + growth_rate_2_to_3)

/-- The total percentage increase from t=0 to t=3 -/
def total_percentage_increase : ℝ := (total_growth_factor - 1) * 100

theorem population_growth_theorem :
  abs (total_percentage_increase - 33.18) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_theorem_l3111_311195


namespace NUMINAMATH_CALUDE_john_hard_hat_ratio_l3111_311112

/-- Proves that the ratio of green to pink hard hats John took away is 2:1 -/
theorem john_hard_hat_ratio :
  let initial_pink : ℕ := 26
  let initial_green : ℕ := 15
  let initial_yellow : ℕ := 24
  let carl_pink : ℕ := 4
  let john_pink : ℕ := 6
  let remaining_total : ℕ := 43
  let initial_total : ℕ := initial_pink + initial_green + initial_yellow
  let john_green : ℕ := initial_total - carl_pink - john_pink - remaining_total
  (john_green : ℚ) / (john_pink : ℚ) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_john_hard_hat_ratio_l3111_311112


namespace NUMINAMATH_CALUDE_point_A_not_on_transformed_plane_l3111_311145

/-- The similarity transformation coefficient -/
def k : ℚ := 2/3

/-- The original plane equation -/
def plane_a (x y z : ℚ) : Prop := 5*x + y - z + 6 = 0

/-- The transformed plane equation -/
def plane_a' (x y z : ℚ) : Prop := 5*x + y - z + 4 = 0

/-- The point A -/
def point_A : ℚ × ℚ × ℚ := (1, -2, 1)

/-- Theorem stating that point A is not on the transformed plane -/
theorem point_A_not_on_transformed_plane : 
  ¬ plane_a' point_A.1 point_A.2.1 point_A.2.2 :=
sorry

end NUMINAMATH_CALUDE_point_A_not_on_transformed_plane_l3111_311145


namespace NUMINAMATH_CALUDE_free_throw_difference_l3111_311190

/-- The number of free-throws made by each player in one minute -/
structure FreeThrows where
  deshawn : ℕ
  kayla : ℕ
  annieka : ℕ

/-- The conditions of the basketball free-throw practice -/
def free_throw_practice (ft : FreeThrows) : Prop :=
  ft.deshawn = 12 ∧
  ft.kayla = ft.deshawn + ft.deshawn / 2 ∧
  ft.annieka = 14 ∧
  ft.annieka < ft.kayla

/-- The theorem stating the difference between Kayla's and Annieka's free-throws -/
theorem free_throw_difference (ft : FreeThrows) 
  (h : free_throw_practice ft) : ft.kayla - ft.annieka = 4 := by
  sorry

#check free_throw_difference

end NUMINAMATH_CALUDE_free_throw_difference_l3111_311190


namespace NUMINAMATH_CALUDE_rational_sum_power_l3111_311138

theorem rational_sum_power (n m : ℚ) (h : (n + 9)^2 + |m - 8| = 0) : 
  (n + m)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_power_l3111_311138


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_range_l3111_311141

theorem quadratic_real_solutions_range (a : ℝ) :
  (∃ x : ℝ, x^2 - 4*x + a = 0) ↔ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_range_l3111_311141


namespace NUMINAMATH_CALUDE_mower_blades_cost_is_47_l3111_311169

/-- The amount Mike made mowing lawns -/
def total_earnings : ℕ := 101

/-- The number of games Mike could buy with the remaining money -/
def num_games : ℕ := 9

/-- The cost of each game -/
def game_cost : ℕ := 6

/-- The amount Mike spent on new mower blades -/
def mower_blades_cost : ℕ := total_earnings - (num_games * game_cost)

theorem mower_blades_cost_is_47 : mower_blades_cost = 47 := by
  sorry

end NUMINAMATH_CALUDE_mower_blades_cost_is_47_l3111_311169
