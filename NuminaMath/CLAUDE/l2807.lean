import Mathlib

namespace NUMINAMATH_CALUDE_min_pet_owners_l2807_280750

/-- Represents the number of people who own only dogs -/
def only_dogs : Nat := 15

/-- Represents the number of people who own only cats -/
def only_cats : Nat := 10

/-- Represents the number of people who own only cats and dogs -/
def cats_and_dogs : Nat := 5

/-- Represents the number of people who own cats, dogs, and snakes -/
def cats_dogs_snakes : Nat := 3

/-- Represents the total number of snakes -/
def total_snakes : Nat := 59

/-- Theorem stating that the minimum number of pet owners is 33 -/
theorem min_pet_owners : 
  only_dogs + only_cats + cats_and_dogs + cats_dogs_snakes = 33 := by
  sorry

#check min_pet_owners

end NUMINAMATH_CALUDE_min_pet_owners_l2807_280750


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2807_280769

theorem fraction_to_decimal : (5 : ℚ) / 8 = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2807_280769


namespace NUMINAMATH_CALUDE_work_earnings_equation_l2807_280794

theorem work_earnings_equation (t : ℝ) : 
  (t + 2) * (4 * t - 4) = (2 * t - 3) * (t + 3) + 3 → 
  t = (-1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_work_earnings_equation_l2807_280794


namespace NUMINAMATH_CALUDE_points_are_coplanar_l2807_280744

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define the points
variable (A B C P O : V)

-- Define the non-collinearity condition
def not_collinear (A B C : V) : Prop :=
  ∀ (t : ℝ), B - A ≠ t • (C - A)

-- Define the vector equation
def vector_equation (O A B C P : V) : Prop :=
  P - O = (3/4) • (A - O) + (1/8) • (B - O) + (1/8) • (C - O)

-- Define coplanarity
def coplanar (A B C P : V) : Prop :=
  ∃ (a b c : ℝ), P - A = a • (B - A) + b • (C - A)

-- State the theorem
theorem points_are_coplanar
  (h1 : not_collinear A B C)
  (h2 : ∀ O, vector_equation O A B C P) :
  coplanar A B C P :=
sorry

end NUMINAMATH_CALUDE_points_are_coplanar_l2807_280744


namespace NUMINAMATH_CALUDE_students_with_two_skills_l2807_280738

theorem students_with_two_skills (total : ℕ) (no_poetry : ℕ) (no_paint : ℕ) (no_instrument : ℕ) 
  (h1 : total = 150)
  (h2 : no_poetry = 80)
  (h3 : no_paint = 90)
  (h4 : no_instrument = 60) :
  let poetry := total - no_poetry
  let paint := total - no_paint
  let instrument := total - no_instrument
  let two_skills := poetry + paint + instrument - total
  two_skills = 70 := by
  sorry

end NUMINAMATH_CALUDE_students_with_two_skills_l2807_280738


namespace NUMINAMATH_CALUDE_hexagon_side_count_l2807_280702

/-- A convex hexagon with exactly two distinct side lengths -/
structure ConvexHexagon where
  side_length1 : ℝ
  side_length2 : ℝ
  num_sides1 : ℕ
  num_sides2 : ℕ
  distinct_lengths : side_length1 ≠ side_length2
  total_sides : num_sides1 + num_sides2 = 6

/-- The perimeter of a convex hexagon -/
def perimeter (h : ConvexHexagon) : ℝ :=
  h.side_length1 * h.num_sides1 + h.side_length2 * h.num_sides2

theorem hexagon_side_count (h : ConvexHexagon)
  (side1_length : h.side_length1 = 8)
  (side2_length : h.side_length2 = 10)
  (total_perimeter : perimeter h = 56) :
  h.num_sides2 = 4 :=
sorry

end NUMINAMATH_CALUDE_hexagon_side_count_l2807_280702


namespace NUMINAMATH_CALUDE_correct_evaluation_l2807_280767

-- Define the expression
def expression : ℤ → ℤ → ℤ → ℤ := λ a b c => a - b * c

-- Define the order of operations
def evaluate_expression (a b c : ℤ) : ℤ :=
  a - (b * c)

-- Theorem statement
theorem correct_evaluation :
  evaluate_expression 65 13 2 = 39 :=
by
  sorry

#eval evaluate_expression 65 13 2

end NUMINAMATH_CALUDE_correct_evaluation_l2807_280767


namespace NUMINAMATH_CALUDE_fourth_power_trinomial_coefficients_l2807_280720

/-- A trinomial that is an exact fourth power for all integers -/
def is_fourth_power (a b c : ℝ) : Prop :=
  ∀ x : ℤ, ∃ y : ℝ, a * x^2 + b * x + c = y^4

/-- If a trinomial is an exact fourth power for all integers, then its quadratic and linear coefficients are zero -/
theorem fourth_power_trinomial_coefficients (a b c : ℝ) :
  is_fourth_power a b c → a = 0 ∧ b = 0 :=
by sorry

end NUMINAMATH_CALUDE_fourth_power_trinomial_coefficients_l2807_280720


namespace NUMINAMATH_CALUDE_l_shaped_tiling_exists_l2807_280778

/-- An L-shaped piece consisting of three squares -/
inductive LPiece
| mk : LPiece

/-- A square grid of side length 2^n -/
def Square (n : ℕ) := Fin (2^n) × Fin (2^n)

/-- A cell in the square grid -/
def Cell (n : ℕ) := Square n

/-- A tiling of the square grid using L-shaped pieces -/
def Tiling (n : ℕ) := Square n → Option LPiece

/-- Predicate to check if a tiling is valid -/
def is_valid_tiling (n : ℕ) (t : Tiling n) (removed : Cell n) : Prop :=
  ∀ (c : Cell n), c ≠ removed → ∃ (piece : LPiece), t c = some piece

/-- The main theorem: for all n, there exists a valid tiling of a 2^n x 2^n square
    with one cell removed using L-shaped pieces -/
theorem l_shaped_tiling_exists (n : ℕ) :
  ∀ (removed : Cell n), ∃ (t : Tiling n), is_valid_tiling n t removed :=
sorry

end NUMINAMATH_CALUDE_l_shaped_tiling_exists_l2807_280778


namespace NUMINAMATH_CALUDE_thirty_three_million_scientific_notation_l2807_280748

/-- Proves that 33 million is equal to 3.3 × 10^7 in scientific notation -/
theorem thirty_three_million_scientific_notation :
  (33 : ℝ) * 1000000 = 3.3 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_thirty_three_million_scientific_notation_l2807_280748


namespace NUMINAMATH_CALUDE_machines_required_scenario2_l2807_280796

-- Define the job completion rate for a single machine in 30 minutes
def single_machine_rate : ℚ := (3 / 4) * (1 / 5)

-- Define the job completion for a single machine in 60 minutes
def single_machine_60min : ℚ := 2 * single_machine_rate

-- Define the number of machines in the first scenario
def machines_scenario1 : ℕ := 5

-- Define the job completion in the first scenario
def job_completion1 : ℚ := 3 / 4

-- Define the time in the first scenario
def time_scenario1 : ℕ := 30

-- Define the job completion in the second scenario
def job_completion2 : ℚ := 3 / 5

-- Define the time in the second scenario
def time_scenario2 : ℕ := 60

-- Theorem to prove
theorem machines_required_scenario2 :
  ∃ (n : ℕ), n * single_machine_60min = job_completion2 ∧ n = 2 := by sorry

end NUMINAMATH_CALUDE_machines_required_scenario2_l2807_280796


namespace NUMINAMATH_CALUDE_book_arrangement_and_distribution_l2807_280758

/-- The number of ways to arrange 5 books, including 2 mathematics books, in a row such that
    the mathematics books are not adjacent and not placed at both ends simultaneously. -/
def arrange_books : ℕ := 60

/-- The number of ways to distribute 5 books, including 2 mathematics books, to 3 students,
    with each student receiving at least 1 book. -/
def distribute_books : ℕ := 150

/-- Theorem stating the correct number of arrangements and distributions -/
theorem book_arrangement_and_distribution :
  arrange_books = 60 ∧ distribute_books = 150 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_and_distribution_l2807_280758


namespace NUMINAMATH_CALUDE_complex_fraction_real_l2807_280781

theorem complex_fraction_real (a : ℝ) : 
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (Complex.ofReal a - Complex.I) / (2 + Complex.I) ∈ Set.range Complex.ofReal →
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l2807_280781


namespace NUMINAMATH_CALUDE_equation_solution_l2807_280725

theorem equation_solution : ∃ x : ℤ, 45 - (28 - (x - (15 - 18))) = 57 ∧ x = 37 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2807_280725


namespace NUMINAMATH_CALUDE_roots_equation_value_l2807_280772

theorem roots_equation_value (α β : ℝ) : 
  α^2 - 2*α - 4 = 0 → β^2 - 2*β - 4 = 0 → α^3 + 8*β + 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_value_l2807_280772


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l2807_280771

/-- A line is tangent to a parabola if and only if their intersection has exactly one point. -/
def is_tangent_line_to_parabola (a b c : ℝ) : Prop :=
  ∃! x : ℝ, (3 * x + 1)^2 = 12 * x

/-- The line y = 3x + 1 is tangent to the parabola y^2 = 12x. -/
theorem line_tangent_to_parabola : is_tangent_line_to_parabola 3 1 12 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l2807_280771


namespace NUMINAMATH_CALUDE_activity_participation_l2807_280708

def total_sample : ℕ := 100
def male_participants : ℕ := 60
def willing_to_participate : ℕ := 70
def males_willing : ℕ := 48
def females_not_willing : ℕ := 18

def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

def critical_value : ℚ := 6635 / 1000

theorem activity_participation :
  let females_willing := willing_to_participate - males_willing
  let males_not_willing := male_participants - males_willing
  let female_participants := total_sample - male_participants
  let chi_sq := chi_square males_willing females_willing males_not_willing females_not_willing
  let male_proportion := (males_willing : ℚ) / male_participants
  let female_proportion := (females_willing : ℚ) / female_participants
  (chi_sq > critical_value) ∧
  (male_proportion > female_proportion) ∧
  (12 / 7 : ℚ) = (4 * 0 + 3 * 1 + 2 * 2 + 1 * 3 : ℚ) / (Nat.choose 7 3) := by sorry

end NUMINAMATH_CALUDE_activity_participation_l2807_280708


namespace NUMINAMATH_CALUDE_vegetable_vendor_calculation_l2807_280745

/-- Represents the vegetable vendor's purchase and sales scenario -/
structure VegetableVendor where
  total_cost : ℝ
  total_weight : ℝ
  cucumber_wholesale_price : ℝ
  cucumber_retail_price : ℝ
  potato_wholesale_price : ℝ
  potato_retail_price : ℝ

/-- Calculates the amount of cucumbers and potatoes purchased and the profit -/
def calculate_purchase_and_profit (v : VegetableVendor) :
  (ℝ × ℝ × ℝ) :=
  let cucumber_kg := 30
  let potato_kg := 10
  let profit := (v.cucumber_retail_price - v.cucumber_wholesale_price) * cucumber_kg +
                (v.potato_retail_price - v.potato_wholesale_price) * potato_kg
  (cucumber_kg, potato_kg, profit)

/-- Theorem stating the correctness of the calculation -/
theorem vegetable_vendor_calculation (v : VegetableVendor)
  (h1 : v.total_cost = 114)
  (h2 : v.total_weight = 40)
  (h3 : v.cucumber_wholesale_price = 3)
  (h4 : v.cucumber_retail_price = 5)
  (h5 : v.potato_wholesale_price = 2.4)
  (h6 : v.potato_retail_price = 4) :
  calculate_purchase_and_profit v = (30, 10, 76) := by
  sorry

end NUMINAMATH_CALUDE_vegetable_vendor_calculation_l2807_280745


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l2807_280739

theorem dining_bill_calculation (total : ℝ) (tip_rate : ℝ) (tax_rate : ℝ) 
  (h1 : total = 132)
  (h2 : tip_rate = 0.20)
  (h3 : tax_rate = 0.10) :
  ∃ (original_price : ℝ), 
    original_price * (1 + tax_rate) * (1 + tip_rate) = total ∧ 
    original_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l2807_280739


namespace NUMINAMATH_CALUDE_arrangement_remainder_l2807_280798

/-- The number of green marbles --/
def green_marbles : ℕ := 7

/-- The maximum number of blue marbles satisfying the arrangement condition --/
def max_blue_marbles : ℕ := 19

/-- The total number of marbles --/
def total_marbles : ℕ := green_marbles + max_blue_marbles

/-- The number of ways to arrange the marbles --/
def arrangement_count : ℕ := Nat.choose total_marbles green_marbles

/-- Theorem stating the remainder when the number of arrangements is divided by 500 --/
theorem arrangement_remainder : arrangement_count % 500 = 30 := by sorry

end NUMINAMATH_CALUDE_arrangement_remainder_l2807_280798


namespace NUMINAMATH_CALUDE_smallest_d_for_inverse_l2807_280721

def g (x : ℝ) : ℝ := (x - 3)^2 + 4

theorem smallest_d_for_inverse (d : ℝ) : 
  (∀ x y, x ∈ Set.Ici d → y ∈ Set.Ici d → g x = g y → x = y) ∧ 
  (∀ d' < d, ∃ x y, x ∈ Set.Ici d' → y ∈ Set.Ici d' → g x = g y ∧ x ≠ y) ↔ 
  d = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_for_inverse_l2807_280721


namespace NUMINAMATH_CALUDE_isosceles_triangle_part1_isosceles_triangle_part2_l2807_280763

/-- Represents an isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  isIsosceles : leg ≥ base
  perimeter : ℝ
  sumOfSides : base + 2 * leg = perimeter

/-- Theorem for part 1 of the problem -/
theorem isosceles_triangle_part1 :
  ∃ (t : IsoscelesTriangle),
    t.perimeter = 20 ∧ t.leg = 2 * t.base ∧ t.base = 4 ∧ t.leg = 8 := by
  sorry

/-- Theorem for part 2 of the problem -/
theorem isosceles_triangle_part2 :
  ∃ (t : IsoscelesTriangle),
    t.perimeter = 20 ∧ t.base = 5 ∧ t.leg = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_part1_isosceles_triangle_part2_l2807_280763


namespace NUMINAMATH_CALUDE_second_number_is_six_l2807_280776

theorem second_number_is_six (x y : ℝ) (h : 3 * y - x = 2 * y + 6) : y = 6 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_six_l2807_280776


namespace NUMINAMATH_CALUDE_base_6_divisibility_l2807_280718

def base_6_to_decimal (y : ℕ) : ℕ := 2 * 6^3 + 4 * 6^2 + y * 6 + 2

def is_valid_base_6_digit (y : ℕ) : Prop := y ≤ 5

theorem base_6_divisibility (y : ℕ) : 
  is_valid_base_6_digit y → (base_6_to_decimal y % 13 = 0 ↔ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_base_6_divisibility_l2807_280718


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l2807_280729

/-- The gain percent when selling a cycle -/
theorem cycle_gain_percent (cost_price selling_price : ℚ) :
  cost_price = 900 →
  selling_price = 1080 →
  (selling_price - cost_price) / cost_price * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l2807_280729


namespace NUMINAMATH_CALUDE_equation_solution_l2807_280735

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (1 - x) / (x - 2) = 1 - 3 / (x - 2) ↔ x = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2807_280735


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l2807_280709

/-- A linear function f(x) = -x + 1 -/
def f (x : ℝ) : ℝ := -x + 1

theorem y1_greater_than_y2 (y₁ y₂ : ℝ) 
  (h₁ : f (-1) = y₁) 
  (h₂ : f 2 = y₂) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l2807_280709


namespace NUMINAMATH_CALUDE_amy_flash_drive_files_l2807_280777

theorem amy_flash_drive_files (initial_music : ℕ) (initial_video : ℕ) (deleted : ℕ) (downloaded : ℕ)
  (h1 : initial_music = 26)
  (h2 : initial_video = 36)
  (h3 : deleted = 48)
  (h4 : downloaded = 15) :
  initial_music + initial_video - deleted + downloaded = 29 := by
  sorry

end NUMINAMATH_CALUDE_amy_flash_drive_files_l2807_280777


namespace NUMINAMATH_CALUDE_cars_without_ac_l2807_280704

/-- Given a group of cars with the following properties:
  * There are 100 cars in total
  * At least 53 cars have racing stripes
  * The greatest number of cars that could have air conditioning but not racing stripes is 47
  Prove that the number of cars without air conditioning is 47. -/
theorem cars_without_ac (total : ℕ) (with_stripes : ℕ) (ac_no_stripes : ℕ)
  (h1 : total = 100)
  (h2 : with_stripes ≥ 53)
  (h3 : ac_no_stripes = 47) :
  total - (ac_no_stripes + (with_stripes - ac_no_stripes)) = 47 := by
  sorry

end NUMINAMATH_CALUDE_cars_without_ac_l2807_280704


namespace NUMINAMATH_CALUDE_salt_mixture_price_l2807_280780

theorem salt_mixture_price (salt_price_1 : ℚ) (salt_weight_1 : ℚ)
  (salt_weight_2 : ℚ) (selling_price : ℚ) (profit_percentage : ℚ) :
  salt_price_1 = 50 / 100 →
  salt_weight_1 = 8 →
  salt_weight_2 = 40 →
  selling_price = 48 / 100 →
  profit_percentage = 20 / 100 →
  ∃ (salt_price_2 : ℚ),
    salt_price_2 * salt_weight_2 + salt_price_1 * salt_weight_1 =
      (selling_price * (salt_weight_1 + salt_weight_2)) / (1 + profit_percentage) ∧
    salt_price_2 = 38 / 100 :=
by sorry

end NUMINAMATH_CALUDE_salt_mixture_price_l2807_280780


namespace NUMINAMATH_CALUDE_stratified_sample_sophomores_l2807_280717

/-- Represents the number of sophomores in a stratified sample -/
def sophomores_in_sample (total_students : ℕ) (total_sophomores : ℕ) (sample_size : ℕ) : ℕ :=
  (sample_size * total_sophomores) / total_students

/-- Theorem: In a school with 1500 students, of which 600 are sophomores,
    a stratified sample of 100 students should include 40 sophomores -/
theorem stratified_sample_sophomores :
  sophomores_in_sample 1500 600 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_sophomores_l2807_280717


namespace NUMINAMATH_CALUDE_article_original_price_l2807_280726

/-- Given an article sold with an 18% profit resulting in a profit of 542.8,
    prove that the original price of the article was 3016. -/
theorem article_original_price (profit_percentage : ℝ) (profit : ℝ) (original_price : ℝ) :
  profit_percentage = 18 →
  profit = 542.8 →
  profit = original_price * (profit_percentage / 100) →
  original_price = 3016 := by
  sorry

end NUMINAMATH_CALUDE_article_original_price_l2807_280726


namespace NUMINAMATH_CALUDE_different_plant_choice_probability_l2807_280752

theorem different_plant_choice_probability :
  let num_plant_types : ℕ := 4
  let num_employees : ℕ := 2
  let total_combinations : ℕ := num_plant_types ^ num_employees
  let same_choice_combinations : ℕ := num_plant_types
  let different_choice_combinations : ℕ := total_combinations - same_choice_combinations
  (different_choice_combinations : ℚ) / total_combinations = 13 / 16 :=
by sorry

end NUMINAMATH_CALUDE_different_plant_choice_probability_l2807_280752


namespace NUMINAMATH_CALUDE_cosine_value_in_triangle_l2807_280732

theorem cosine_value_in_triangle (a b c : ℝ) (h : 3 * a^2 + 3 * b^2 - 3 * c^2 = 2 * a * b) :
  let cosC := (a^2 + b^2 - c^2) / (2 * a * b)
  cosC = 1/3 := by sorry

end NUMINAMATH_CALUDE_cosine_value_in_triangle_l2807_280732


namespace NUMINAMATH_CALUDE_alice_purse_value_l2807_280759

-- Define the values of coins in cents
def penny : ℕ := 1
def dime : ℕ := 10
def quarter : ℕ := 25
def half_dollar : ℕ := 50

-- Define the total value of coins in Alice's purse
def purse_value : ℕ := penny + dime + quarter + half_dollar

-- Define one dollar in cents
def one_dollar : ℕ := 100

-- Theorem statement
theorem alice_purse_value :
  (purse_value : ℚ) / one_dollar = 86 / 100 := by sorry

end NUMINAMATH_CALUDE_alice_purse_value_l2807_280759


namespace NUMINAMATH_CALUDE_statement_1_statement_3_statement_4_l2807_280770

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (skew_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- Define the planes and lines
variable (a b : Plane)
variable (l m n : Line)

-- Define the non-coincidence conditions
variable (planes_non_coincident : a ≠ b)
variable (lines_non_coincident : l ≠ m ∧ m ≠ n ∧ l ≠ n)

-- Theorem for statement 1
theorem statement_1 :
  parallel_planes a b →
  line_in_plane l a →
  parallel_line_plane l b :=
sorry

-- Theorem for statement 3
theorem statement_3 :
  parallel_line_plane l a →
  perpendicular_line_plane l b →
  perpendicular_planes a b :=
sorry

-- Theorem for statement 4
theorem statement_4 :
  skew_lines m n →
  parallel_line_plane m a →
  parallel_line_plane n a →
  perpendicular_lines l m →
  perpendicular_lines l n →
  perpendicular_line_plane l a :=
sorry

end NUMINAMATH_CALUDE_statement_1_statement_3_statement_4_l2807_280770


namespace NUMINAMATH_CALUDE_distance_to_school_l2807_280787

/-- The distance to school given the travel conditions -/
theorem distance_to_school (total_time : ℝ) (speed_to_school : ℝ) (speed_from_school : ℝ) 
  (h1 : total_time = 1)
  (h2 : speed_to_school = 5)
  (h3 : speed_from_school = 21) :
  ∃ d : ℝ, d = 105 / 26 ∧ d / speed_to_school + d / speed_from_school = total_time := by
  sorry

end NUMINAMATH_CALUDE_distance_to_school_l2807_280787


namespace NUMINAMATH_CALUDE_total_chickens_and_ducks_l2807_280706

theorem total_chickens_and_ducks (num_chickens : ℕ) (duck_difference : ℕ) : 
  num_chickens = 45 → 
  duck_difference = 8 → 
  num_chickens + (num_chickens - duck_difference) = 82 :=
by sorry

end NUMINAMATH_CALUDE_total_chickens_and_ducks_l2807_280706


namespace NUMINAMATH_CALUDE_fun_math_book_price_l2807_280701

/-- The price of the "Fun Math" book in yuan -/
def book_price : ℝ := 4

/-- The amount Xiaohong is short in yuan -/
def xiaohong_short : ℝ := 2.2

/-- The amount Xiaoming is short in yuan -/
def xiaoming_short : ℝ := 1.8

/-- Theorem stating that the book price is 4 yuan given the conditions -/
theorem fun_math_book_price :
  (book_price - xiaohong_short) + (book_price - xiaoming_short) = book_price :=
by sorry

end NUMINAMATH_CALUDE_fun_math_book_price_l2807_280701


namespace NUMINAMATH_CALUDE_chopping_percentage_difference_l2807_280743

/-- Represents the chopping rates and total amount for Tom and Tammy -/
structure ChoppingData where
  tom_rate : ℚ  -- Tom's chopping rate in lb/min
  tammy_rate : ℚ  -- Tammy's chopping rate in lb/min
  total_amount : ℚ  -- Total amount of salad chopped in lb

/-- Calculates the percentage difference between Tammy's and Tom's chopped quantities -/
def percentage_difference (data : ChoppingData) : ℚ :=
  let combined_rate := data.tom_rate + data.tammy_rate
  let tom_share := (data.tom_rate / combined_rate) * data.total_amount
  let tammy_share := (data.tammy_rate / combined_rate) * data.total_amount
  ((tammy_share - tom_share) / tom_share) * 100

/-- Theorem stating that the percentage difference is 125% for the given data -/
theorem chopping_percentage_difference :
  let data : ChoppingData := {
    tom_rate := 2 / 3,  -- 2 lb in 3 minutes
    tammy_rate := 3 / 2,  -- 3 lb in 2 minutes
    total_amount := 65
  }
  percentage_difference data = 125 := by sorry


end NUMINAMATH_CALUDE_chopping_percentage_difference_l2807_280743


namespace NUMINAMATH_CALUDE_sons_age_l2807_280736

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 28 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 26 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2807_280736


namespace NUMINAMATH_CALUDE_expected_balls_in_original_position_l2807_280737

/-- Represents the number of balls in the circle. -/
def n : ℕ := 6

/-- Represents the number of swaps performed. -/
def k : ℕ := 3

/-- The probability of a specific ball being swapped in one swap. -/
def p : ℚ := 1 / 3

/-- The probability of a ball remaining in its original position after k swaps. -/
def prob_original_position (k : ℕ) (p : ℚ) : ℚ :=
  (1 - p)^k + k * p * (1 - p)^(k-1)

/-- The expected number of balls in their original positions after k swaps. -/
def expected_original_positions (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  n * prob_original_position k p

theorem expected_balls_in_original_position :
  expected_original_positions n k p = 84 / 27 :=
by sorry

end NUMINAMATH_CALUDE_expected_balls_in_original_position_l2807_280737


namespace NUMINAMATH_CALUDE_solution_sum_l2807_280799

-- Define the solution set for |2x-3| ≤ 1
def solution_set (m n : ℝ) : Prop :=
  ∀ x, |2*x - 3| ≤ 1 ↔ m ≤ x ∧ x ≤ n

-- Theorem statement
theorem solution_sum (m n : ℝ) : solution_set m n → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_sum_l2807_280799


namespace NUMINAMATH_CALUDE_fifth_number_pascal_proof_l2807_280786

/-- The fifth number in the row of Pascal's triangle that starts with 1 and 15 -/
def fifth_number_pascal : ℕ := 1365

/-- The row of Pascal's triangle we're interested in -/
def pascal_row : List ℕ := [1, 15]

/-- Theorem stating that the fifth number in the specified row of Pascal's triangle is 1365 -/
theorem fifth_number_pascal_proof : 
  ∀ (row : List ℕ), row = pascal_row → 
  (List.nthLe row 4 sorry : ℕ) = fifth_number_pascal := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_pascal_proof_l2807_280786


namespace NUMINAMATH_CALUDE_solution_set_f_min_value_fraction_equality_condition_l2807_280774

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + 2 * |x - 1|

-- Theorem for the solution set of f(x) ≤ 4
theorem solution_set_f :
  {x : ℝ | f x ≤ 4} = {x : ℝ | -1 ≤ x ∧ x ≤ 5/3} :=
sorry

-- Theorem for the minimum value of 2/a + 1/b
theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 4) :
  2/a + 1/b ≥ 2 :=
sorry

-- Theorem for the equality condition
theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 4) :
  2/a + 1/b = 2 ↔ a = 2 ∧ b = 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_min_value_fraction_equality_condition_l2807_280774


namespace NUMINAMATH_CALUDE_quadruple_solution_l2807_280719

theorem quadruple_solution :
  ∀ (a b c d : ℝ), 
    a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
    a^2 * b = c →
    b * c^2 = a →
    c * a^2 = b →
    a + b + c = d →
    a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadruple_solution_l2807_280719


namespace NUMINAMATH_CALUDE_A_minus_2B_A_minus_2B_special_case_A_minus_2B_independent_of_x_l2807_280714

-- Define the expressions A and B
def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y
def B (x y : ℝ) : ℝ := x^2 - x * y + x

-- Theorem 1: A - 2B = 5xy - 2x + 2y
theorem A_minus_2B (x y : ℝ) : A x y - 2 * B x y = 5 * x * y - 2 * x + 2 * y := by sorry

-- Theorem 2: When x² = 9 and |y| = 2, A - 2B ∈ {28, -40, -20, 32}
theorem A_minus_2B_special_case (x y : ℝ) (h1 : x^2 = 9) (h2 : |y| = 2) :
  A x y - 2 * B x y ∈ ({28, -40, -20, 32} : Set ℝ) := by sorry

-- Theorem 3: If A - 2B is independent of x, then y = 2/5
theorem A_minus_2B_independent_of_x (y : ℝ) :
  (∀ x : ℝ, A x y - 2 * B x y = A 0 y - 2 * B 0 y) → y = 2/5 := by sorry

end NUMINAMATH_CALUDE_A_minus_2B_A_minus_2B_special_case_A_minus_2B_independent_of_x_l2807_280714


namespace NUMINAMATH_CALUDE_sausage_distance_ratio_l2807_280722

/-- Represents the scenario of a dog and cat running towards sausages --/
structure SausageScenario where
  dog_speed : ℝ
  cat_speed : ℝ
  dog_eat_rate : ℝ
  cat_eat_rate : ℝ
  total_sausages : ℝ
  total_distance : ℝ

/-- The theorem to be proved --/
theorem sausage_distance_ratio 
  (scenario : SausageScenario)
  (h1 : scenario.cat_speed = 2 * scenario.dog_speed)
  (h2 : scenario.dog_eat_rate = scenario.cat_eat_rate / 2)
  (h3 : scenario.cat_eat_rate * 1 = scenario.total_sausages)
  (h4 : scenario.cat_speed * 1 = scenario.total_distance)
  (h5 : scenario.total_sausages > 0)
  (h6 : scenario.total_distance > 0) :
  ∃ (cat_distance dog_distance : ℝ),
    cat_distance + dog_distance = scenario.total_distance ∧
    cat_distance / dog_distance = 7 / 5 := by
  sorry


end NUMINAMATH_CALUDE_sausage_distance_ratio_l2807_280722


namespace NUMINAMATH_CALUDE_valid_factorization_l2807_280791

theorem valid_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_valid_factorization_l2807_280791


namespace NUMINAMATH_CALUDE_sqrt_over_thirteen_equals_four_l2807_280756

theorem sqrt_over_thirteen_equals_four :
  Real.sqrt 2704 / 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_over_thirteen_equals_four_l2807_280756


namespace NUMINAMATH_CALUDE_min_value_theorem_l2807_280760

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x + 2*y)) + (y / x) ≥ Real.sqrt 2 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2807_280760


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2807_280788

/-- Given a stratified sample with ratio 2:3:5 for products A:B:C, 
    prove that if 16 type A products are sampled, the total sample size is 80 -/
theorem stratified_sample_size 
  (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ) 
  (h_ratio : ratio_A = 2 ∧ ratio_B = 3 ∧ ratio_C = 5) 
  (sample_A : ℕ) (h_sample_A : sample_A = 16) : 
  let total_ratio := ratio_A + ratio_B + ratio_C
  let sample_size := (sample_A * total_ratio) / ratio_A
  sample_size = 80 := by sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2807_280788


namespace NUMINAMATH_CALUDE_angle_B_measure_l2807_280773

-- Define the hexagon NUMBERS
structure Hexagon :=
  (N U M B E S : ℝ)

-- Define the properties of the hexagon
def is_valid_hexagon (h : Hexagon) : Prop :=
  h.N + h.U + h.M + h.B + h.E + h.S = 720 ∧ 
  h.N = h.M ∧ h.M = h.B ∧
  h.U + h.S = 180

-- Theorem statement
theorem angle_B_measure (h : Hexagon) (hvalid : is_valid_hexagon h) : h.B = 135 := by
  sorry


end NUMINAMATH_CALUDE_angle_B_measure_l2807_280773


namespace NUMINAMATH_CALUDE_day_of_week_previous_year_l2807_280785

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ
  isLeap : Bool

/-- Returns the day of the week given a day number and a year -/
def dayOfWeek (day : ℕ) (year : Year) : DayOfWeek :=
  sorry

/-- Advances the day of the week by a given number of days -/
def advanceDays (start : DayOfWeek) (days : ℕ) : DayOfWeek :=
  sorry

theorem day_of_week_previous_year 
  (N : Year)
  (h1 : N.isLeap = true)
  (h2 : dayOfWeek 250 N = DayOfWeek.Wednesday)
  (h3 : dayOfWeek 150 ⟨N.value + 1, false⟩ = DayOfWeek.Wednesday) :
  dayOfWeek 100 ⟨N.value - 1, false⟩ = DayOfWeek.Saturday :=
by sorry

end NUMINAMATH_CALUDE_day_of_week_previous_year_l2807_280785


namespace NUMINAMATH_CALUDE_evaluate_expression_l2807_280755

theorem evaluate_expression : 3 * 403 + 5 * 403 + 2 * 403 + 401 = 4431 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2807_280755


namespace NUMINAMATH_CALUDE_cylinder_volume_relationship_l2807_280793

/-- Theorem about the volumes of two cylinders with specific relationships -/
theorem cylinder_volume_relationship (h r_C r_D h_C h_D : ℝ) 
  (h_positive : h > 0)
  (cylinder_C : r_C = h ∧ h_C = 3 * r_D)
  (cylinder_D : r_D = h / 3 ∧ h_D = h)
  (volume_ratio : π * r_D^2 * h_D = 3 * (π * r_C^2 * h_C)) :
  π * r_D^2 * h_D = 3 * π * h^3 :=
sorry

end NUMINAMATH_CALUDE_cylinder_volume_relationship_l2807_280793


namespace NUMINAMATH_CALUDE_circular_arrangement_theorem_l2807_280733

/-- Represents a circular seating arrangement of men and women -/
structure CircularArrangement where
  total_people : ℕ
  women : ℕ
  men : ℕ
  women_left_of_women : ℕ
  men_left_of_women : ℕ
  women_right_of_men_ratio : ℚ

/-- The properties of the circular arrangement in the problem -/
def problem_arrangement : CircularArrangement where
  total_people := 35
  women := 19
  men := 16
  women_left_of_women := 7
  men_left_of_women := 12
  women_right_of_men_ratio := 3/4

theorem circular_arrangement_theorem (arr : CircularArrangement) :
  arr.women_left_of_women = 7 ∧
  arr.men_left_of_women = 12 ∧
  arr.women_right_of_men_ratio = 3/4 →
  arr.total_people = 35 ∧
  arr.women = 19 ∧
  arr.men = 16 := by
  sorry

#check circular_arrangement_theorem problem_arrangement

end NUMINAMATH_CALUDE_circular_arrangement_theorem_l2807_280733


namespace NUMINAMATH_CALUDE_students_correct_both_experiments_l2807_280710

/-- Given a group of students performing physics and chemistry experiments, 
    calculate the number of students who conducted both experiments correctly. -/
theorem students_correct_both_experiments 
  (total : ℕ) 
  (physics_correct : ℕ) 
  (chemistry_correct : ℕ) 
  (both_incorrect : ℕ) 
  (h1 : total = 50)
  (h2 : physics_correct = 40)
  (h3 : chemistry_correct = 31)
  (h4 : both_incorrect = 5) :
  physics_correct + chemistry_correct + both_incorrect - total = 26 := by
  sorry

#eval 40 + 31 + 5 - 50  -- Should output 26

end NUMINAMATH_CALUDE_students_correct_both_experiments_l2807_280710


namespace NUMINAMATH_CALUDE_smallest_collection_l2807_280740

def yoongi_collection : ℕ := 4
def yuna_collection : ℕ := 5
def jungkook_collection : ℕ := 6 + 3

theorem smallest_collection : 
  yoongi_collection < yuna_collection ∧ 
  yoongi_collection < jungkook_collection := by
sorry

end NUMINAMATH_CALUDE_smallest_collection_l2807_280740


namespace NUMINAMATH_CALUDE_square_with_external_triangle_l2807_280746

/-- Given a square ABCD with side length s and an equilateral triangle CDE
    constructed externally on side CD, the ratio of AE to AB is 1 + √3/2 -/
theorem square_with_external_triangle (s : ℝ) (s_pos : s > 0) :
  let AB := s
  let AD := s
  let CD := s
  let CE := s
  let DE := s
  let CDE_altitude := s * Real.sqrt 3 / 2
  let AE := AD + CDE_altitude
  AE / AB = 1 + Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_square_with_external_triangle_l2807_280746


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l2807_280742

/-- Represents a square quilt block -/
structure QuiltBlock where
  size : Nat
  full_shaded : Nat
  half_shaded : Nat

/-- Calculates the fraction of shaded area in a quilt block -/
def shaded_fraction (q : QuiltBlock) : Rat :=
  (q.full_shaded + q.half_shaded / 2 : Rat) / (q.size * q.size)

/-- Theorem stating that a 4x4 quilt block with 2 fully shaded squares and 4 half-shaded squares has 1/4 of its area shaded -/
theorem quilt_shaded_fraction :
  let q : QuiltBlock := { size := 4, full_shaded := 2, half_shaded := 4 }
  shaded_fraction q = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l2807_280742


namespace NUMINAMATH_CALUDE_cot_sixty_degrees_l2807_280789

theorem cot_sixty_degrees : Real.cos (π / 3) / Real.sin (π / 3) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_sixty_degrees_l2807_280789


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l2807_280783

/-- The size of the square grid -/
def gridSize : ℕ := 6

/-- The number of shaded squares -/
def shadedSquares : ℕ := 16

/-- The total number of squares in the grid -/
def totalSquares : ℕ := gridSize * gridSize

/-- The percentage of shaded area -/
def shadedPercentage : ℚ := (shadedSquares : ℚ) / (totalSquares : ℚ) * 100

theorem shaded_area_percentage :
  shadedPercentage = 4444 / 10000 := by sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l2807_280783


namespace NUMINAMATH_CALUDE_circle_equation_l2807_280734

theorem circle_equation (x y k : ℝ) : 
  (∃ h c : ℝ, ∀ x y, x^2 + 14*x + y^2 + 8*y - k = 0 ↔ (x - h)^2 + (y - c)^2 = 10^2) ↔ 
  k = 35 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l2807_280734


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l2807_280797

theorem cos_2alpha_value (α : Real) 
  (h : (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1/2) : 
  Real.cos (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l2807_280797


namespace NUMINAMATH_CALUDE_annas_money_l2807_280741

theorem annas_money (original : ℝ) : 
  (original - original * (1/4) = 24) → original = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_annas_money_l2807_280741


namespace NUMINAMATH_CALUDE_simplify_expression_l2807_280779

theorem simplify_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b < 0) (hc : c < 0) 
  (hab : abs a > abs b) (hca : abs c > abs a) : 
  abs (a + c) - abs (b + c) - abs (a + b) = -2 * a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2807_280779


namespace NUMINAMATH_CALUDE_alices_number_l2807_280754

theorem alices_number : ∃ n : ℕ, 
  (180 ∣ n) ∧ 
  (45 ∣ n) ∧ 
  1000 ≤ n ∧ 
  n < 3000 ∧ 
  (∀ m : ℕ, (180 ∣ m) ∧ (45 ∣ m) ∧ 1000 ≤ m ∧ m < 3000 → n ≤ m) ∧
  n = 1260 :=
by sorry

end NUMINAMATH_CALUDE_alices_number_l2807_280754


namespace NUMINAMATH_CALUDE_unique_true_proposition_l2807_280761

theorem unique_true_proposition :
  (¬ ∀ x : ℝ, x^2 + 3 < 0) ∧
  (¬ ∀ x : ℕ, x^2 ≥ 1) ∧
  (∃ x : ℤ, x^5 < 1) ∧
  (¬ ∃ x : ℚ, x^2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_true_proposition_l2807_280761


namespace NUMINAMATH_CALUDE_cubic_root_sum_ninth_power_l2807_280768

theorem cubic_root_sum_ninth_power (u v w : ℂ) : 
  (u^3 - 3*u - 1 = 0) → 
  (v^3 - 3*v - 1 = 0) → 
  (w^3 - 3*w - 1 = 0) → 
  u^9 + v^9 + w^9 = 246 := by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_ninth_power_l2807_280768


namespace NUMINAMATH_CALUDE_sum_of_distances_l2807_280727

/-- Parabola with equation y^2 = 4x -/
structure Parabola where
  eq : ∀ x y : ℝ, y^2 = 4*x

/-- Line with equation 2x + y - 4 = 0 -/
structure Line where
  eq : ∀ x y : ℝ, 2*x + y - 4 = 0

/-- Point A with coordinates (1, 2) -/
def A : ℝ × ℝ := (1, 2)

/-- Point B, the other intersection of the parabola and line -/
def B : ℝ × ℝ := sorry

/-- F is the focus of the parabola -/
def F : ℝ × ℝ := sorry

/-- |FA| is the distance between F and A -/
def FA : ℝ := sorry

/-- |FB| is the distance between F and B -/
def FB : ℝ := sorry

/-- Theorem stating that |FA| + |FB| = 7 -/
theorem sum_of_distances (p : Parabola) (l : Line) : FA + FB = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distances_l2807_280727


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_157_l2807_280715

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List ℕ := sorry

/-- Sums the digits of a natural number in base 10 -/
def sumDigits (n : ℕ) : ℕ := sorry

/-- Sums the elements of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ := sorry

theorem sum_of_binary_digits_157 : 
  let binary157 := toBinary 157
  let sumBinary157 := sumList binary157
  let sumDigits157 := sumDigits 157
  let binarySumDigits157 := toBinary sumDigits157
  let sumBinarySumDigits157 := sumList binarySumDigits157
  sumBinary157 + sumBinarySumDigits157 = 8 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_157_l2807_280715


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l2807_280784

theorem fifteenth_student_age 
  (total_students : ℕ)
  (avg_age_all : ℝ)
  (num_group1 : ℕ)
  (avg_age_group1 : ℝ)
  (num_group2 : ℕ)
  (avg_age_group2 : ℝ)
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : num_group1 = 5)
  (h4 : avg_age_group1 = 14)
  (h5 : num_group2 = 9)
  (h6 : avg_age_group2 = 16)
  (h7 : num_group1 + num_group2 + 1 = total_students) :
  (total_students : ℝ) * avg_age_all - 
  ((num_group1 : ℝ) * avg_age_group1 + (num_group2 : ℝ) * avg_age_group2) = 11 :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l2807_280784


namespace NUMINAMATH_CALUDE_vip_seat_cost_l2807_280764

theorem vip_seat_cost (total_tickets : ℕ) (total_revenue : ℕ) 
  (general_price : ℕ) (vip_difference : ℕ) :
  total_tickets = 320 →
  total_revenue = 7500 →
  general_price = 15 →
  vip_difference = 212 →
  ∃ (vip_price : ℕ), 
    vip_price = 65 ∧
    (total_tickets - vip_difference) * general_price + 
    vip_difference * vip_price = total_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_vip_seat_cost_l2807_280764


namespace NUMINAMATH_CALUDE_min_value_of_function_l2807_280749

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  x + 2 / (2 * x + 1) - 3 / 2 ≥ 0 ∧ ∃ y > 0, y + 2 / (2 * y + 1) - 3 / 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2807_280749


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2807_280747

/-- Tetrahedron PQRS with given properties -/
structure Tetrahedron where
  -- Edge length QR
  qr : ℝ
  -- Area of face PQR
  area_pqr : ℝ
  -- Area of face QRS
  area_qrs : ℝ
  -- Angle between faces PQR and QRS (in radians)
  angle_pqr_qrs : ℝ

/-- The volume of the tetrahedron PQRS -/
def tetrahedron_volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the specific tetrahedron -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    qr := 15,
    area_pqr := 150,
    area_qrs := 90,
    angle_pqr_qrs := π / 4  -- 45° in radians
  }
  tetrahedron_volume t = 300 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2807_280747


namespace NUMINAMATH_CALUDE_smallest_dual_representation_l2807_280775

/-- Represents a number in a given base with repeated digits -/
def repeatedDigitNumber (digit : Nat) (base : Nat) : Nat :=
  digit * base + digit

/-- Checks if a digit is valid in a given base -/
def isValidDigit (digit : Nat) (base : Nat) : Prop :=
  digit < base

theorem smallest_dual_representation : ∃ (n : Nat),
  (∃ (A : Nat), isValidDigit A 5 ∧ n = repeatedDigitNumber A 5) ∧
  (∃ (B : Nat), isValidDigit B 7 ∧ n = repeatedDigitNumber B 7) ∧
  (∀ (m : Nat),
    ((∃ (A : Nat), isValidDigit A 5 ∧ m = repeatedDigitNumber A 5) ∧
     (∃ (B : Nat), isValidDigit B 7 ∧ m = repeatedDigitNumber B 7))
    → m ≥ n) ∧
  n = 24 :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_representation_l2807_280775


namespace NUMINAMATH_CALUDE_unique_valid_grid_l2807_280782

/-- Represents a 3x3 grid with letters A, B, and C -/
def Grid := Fin 3 → Fin 3 → Fin 3

/-- Checks if a row contains exactly one of each letter -/
def valid_row (g : Grid) (row : Fin 3) : Prop :=
  ∀ letter : Fin 3, ∃! col : Fin 3, g row col = letter

/-- Checks if a column contains exactly one of each letter -/
def valid_column (g : Grid) (col : Fin 3) : Prop :=
  ∀ letter : Fin 3, ∃! row : Fin 3, g row col = letter

/-- Checks if the primary diagonal contains exactly one of each letter -/
def valid_diagonal (g : Grid) : Prop :=
  ∀ letter : Fin 3, ∃! i : Fin 3, g i i = letter

/-- Checks if A is in the upper left corner -/
def a_in_corner (g : Grid) : Prop := g 0 0 = 0

/-- Checks if the grid is valid according to all conditions -/
def valid_grid (g : Grid) : Prop :=
  (∀ row : Fin 3, valid_row g row) ∧
  (∀ col : Fin 3, valid_column g col) ∧
  valid_diagonal g ∧
  a_in_corner g

/-- The main theorem: there is exactly one valid grid arrangement -/
theorem unique_valid_grid : ∃! g : Grid, valid_grid g :=
  sorry

end NUMINAMATH_CALUDE_unique_valid_grid_l2807_280782


namespace NUMINAMATH_CALUDE_expand_cubic_sum_product_l2807_280707

theorem expand_cubic_sum_product (x : ℝ) : (x^3 + 3) * (x^3 + 4) = x^6 + 7*x^3 + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_cubic_sum_product_l2807_280707


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2807_280730

theorem quadratic_equation_solution :
  ∃ x : ℝ, 4 * x^2 - 12 * x + 9 = 0 ∧ x = 3/2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2807_280730


namespace NUMINAMATH_CALUDE_swimming_improvement_l2807_280765

-- Define the initial performance
def initial_laps : ℕ := 15
def initial_time : ℕ := 30

-- Define the improved performance
def improved_laps : ℕ := 20
def improved_time : ℕ := 36

-- Define the improvement in lap time
def lap_time_improvement : ℚ := 
  (initial_time : ℚ) / initial_laps - (improved_time : ℚ) / improved_laps

-- Theorem statement
theorem swimming_improvement : lap_time_improvement = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_swimming_improvement_l2807_280765


namespace NUMINAMATH_CALUDE_solve_system_l2807_280705

theorem solve_system (x y : ℝ) (h1 : 3 * x + y = 75) (h2 : 2 * (3 * x + y) - y = 138) : x = 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2807_280705


namespace NUMINAMATH_CALUDE_charles_papers_left_l2807_280795

/-- The number of papers Charles initially bought -/
def initial_papers : ℕ := 20

/-- The number of pictures Charles drew today -/
def pictures_today : ℕ := 6

/-- The number of pictures Charles drew before work yesterday -/
def pictures_yesterday_before : ℕ := 6

/-- The number of pictures Charles drew after work yesterday -/
def pictures_yesterday_after : ℕ := 6

/-- The number of papers Charles has left -/
def papers_left : ℕ := initial_papers - (pictures_today + pictures_yesterday_before + pictures_yesterday_after)

theorem charles_papers_left : papers_left = 2 := by
  sorry

end NUMINAMATH_CALUDE_charles_papers_left_l2807_280795


namespace NUMINAMATH_CALUDE_no_solution_exists_l2807_280712

theorem no_solution_exists : ¬∃ (x : ℝ), 3 * (2*x)^2 - 2 * (2*x) + 5 = 2 * (6*x^2 - 3*(2*x) + 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2807_280712


namespace NUMINAMATH_CALUDE_brick_prism_surface_area_l2807_280700

theorem brick_prism_surface_area :
  ∀ (a b h : ℝ),
    a > 0 ∧ b > 0 ∧ h > 0 →
    a = 4 * h →
    2 * a = 3 * b →
    a * b * h = 288 →
    2 * ((2 * a) * a + (2 * a) * (b + h) + a * (b + h)) = 1368 :=
by
  sorry

end NUMINAMATH_CALUDE_brick_prism_surface_area_l2807_280700


namespace NUMINAMATH_CALUDE_natalia_albums_count_l2807_280724

/-- Represents the number of items in Natalia's library --/
structure LibraryItems where
  novels : Nat
  comics : Nat
  documentaries : Nat
  albums : Nat

/-- Represents the crate information --/
structure CrateInfo where
  capacity : Nat
  count : Nat

/-- Theorem: Given the library items and crate information, prove that Natalia has 209 albums --/
theorem natalia_albums_count
  (items : LibraryItems)
  (crates : CrateInfo)
  (h1 : items.novels = 145)
  (h2 : items.comics = 271)
  (h3 : items.documentaries = 419)
  (h4 : crates.capacity = 9)
  (h5 : crates.count = 116)
  (h6 : items.novels + items.comics + items.documentaries + items.albums = crates.capacity * crates.count) :
  items.albums = 209 := by
  sorry


end NUMINAMATH_CALUDE_natalia_albums_count_l2807_280724


namespace NUMINAMATH_CALUDE_toms_running_days_l2807_280792

/-- Proves that Tom runs 5 days a week given his running schedule and total distance covered -/
theorem toms_running_days 
  (hours_per_day : ℝ) 
  (speed : ℝ) 
  (total_miles_per_week : ℝ) 
  (h1 : hours_per_day = 1.5)
  (h2 : speed = 8)
  (h3 : total_miles_per_week = 60) :
  (total_miles_per_week / (speed * hours_per_day)) = 5 := by
  sorry


end NUMINAMATH_CALUDE_toms_running_days_l2807_280792


namespace NUMINAMATH_CALUDE_binomial_expansion_terms_l2807_280713

theorem binomial_expansion_terms (x a : ℝ) (n : ℕ) : 
  (Nat.choose n 1 * x^(n-1) * a = 56) →
  (Nat.choose n 2 * x^(n-2) * a^2 = 168) →
  (Nat.choose n 3 * x^(n-3) * a^3 = 336) →
  n = 5 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_terms_l2807_280713


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l2807_280731

/-- The product of the given numbers -/
def product : ℕ := 101 * 103 * 105 * 107

/-- The set of prime factors of the product -/
def prime_factors : Finset ℕ := sorry

theorem distinct_prime_factors_count :
  Finset.card prime_factors = 6 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l2807_280731


namespace NUMINAMATH_CALUDE_shaded_triangle_probability_l2807_280711

/-- Given a set of triangles, some of which are shaded, this theorem proves
    the probability of selecting a shaded triangle. -/
theorem shaded_triangle_probability
  (total_triangles : ℕ)
  (shaded_triangles : ℕ)
  (h1 : total_triangles = 6)
  (h2 : shaded_triangles = 3)
  (h3 : shaded_triangles ≤ total_triangles)
  (h4 : total_triangles > 0) :
  (shaded_triangles : ℚ) / total_triangles = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_shaded_triangle_probability_l2807_280711


namespace NUMINAMATH_CALUDE_largest_common_term_of_arithmetic_progressions_l2807_280723

theorem largest_common_term_of_arithmetic_progressions :
  let seq1 (n : ℕ) := 4 + 5 * n
  let seq2 (m : ℕ) := 3 + 7 * m
  ∃ (n m : ℕ), seq1 n = seq2 m ∧ seq1 n = 299 ∧
  ∀ (k l : ℕ), seq1 k = seq2 l → seq1 k ≤ 299 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_of_arithmetic_progressions_l2807_280723


namespace NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l2807_280790

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- 15! ends with 3 zeroes when written in base 8 -/
theorem fifteen_factorial_base_eight_zeroes :
  trailingZeroes 15 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l2807_280790


namespace NUMINAMATH_CALUDE_rotation_result_l2807_280766

-- Define the shapes
inductive Shape
| Triangle
| Circle
| Square

-- Define the position of shapes in the figure
structure Figure :=
(pos1 : Shape)
(pos2 : Shape)
(pos3 : Shape)

-- Define the rotation operation
def rotate120 (f : Figure) : Figure :=
{ pos1 := f.pos3,
  pos2 := f.pos1,
  pos3 := f.pos2 }

-- Theorem statement
theorem rotation_result (f : Figure) 
  (h1 : f.pos1 ≠ f.pos2) 
  (h2 : f.pos2 ≠ f.pos3) 
  (h3 : f.pos3 ≠ f.pos1) : 
  rotate120 f = 
  { pos1 := f.pos3,
    pos2 := f.pos1,
    pos3 := f.pos2 } := by
  sorry

#check rotation_result

end NUMINAMATH_CALUDE_rotation_result_l2807_280766


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l2807_280716

theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
  (3 / 4 : ℚ) * 16 * banana_value = 6 * orange_value →
  (1 / 3 : ℚ) * 9 * banana_value = (3 / 2 : ℚ) * orange_value :=
by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l2807_280716


namespace NUMINAMATH_CALUDE_optimal_move_is_MN_l2807_280753

/-- Represents a move in the game -/
inductive Move
| FG
| MN

/-- Represents the outcome of the game -/
structure Outcome :=
(player_score : ℕ)
(opponent_score : ℕ)

/-- The game state after 12 moves (6 by each player) -/
def initial_state : ℕ := 12

/-- Simulates the game outcome based on the chosen move -/
def simulate_game (move : Move) : Outcome :=
  match move with
  | Move.FG => ⟨1, 8⟩
  | Move.MN => ⟨8, 1⟩

/-- Determines if one outcome is better than another for the player -/
def is_better_outcome (o1 o2 : Outcome) : Prop :=
  o1.player_score > o2.player_score

theorem optimal_move_is_MN :
  let fg_outcome := simulate_game Move.FG
  let mn_outcome := simulate_game Move.MN
  is_better_outcome mn_outcome fg_outcome :=
by sorry


end NUMINAMATH_CALUDE_optimal_move_is_MN_l2807_280753


namespace NUMINAMATH_CALUDE_field_area_in_square_yards_l2807_280751

/-- Conversion rate from feet to yards -/
def feet_to_yard : ℝ := 3

/-- Length of the field in feet -/
def field_length_feet : ℝ := 12

/-- Width of the field in feet -/
def field_width_feet : ℝ := 9

/-- Theorem stating that the area of the field in square yards is 12 -/
theorem field_area_in_square_yards :
  (field_length_feet / feet_to_yard) * (field_width_feet / feet_to_yard) = 12 :=
by sorry

end NUMINAMATH_CALUDE_field_area_in_square_yards_l2807_280751


namespace NUMINAMATH_CALUDE_sms_genuine_iff_criteria_sms_scam_iff_not_genuine_l2807_280728

/-- Represents an SMS message -/
structure SMS where
  sender : Nat
  content : String

/-- Represents a bank -/
structure Bank where
  name : String
  officialSMSNumber : Nat
  customerServiceNumber : Nat

/-- Predicate to check if an SMS is genuine -/
def is_genuine_sms (s : SMS) (b : Bank) : Prop :=
  s.sender = b.officialSMSNumber ∧
  ∃ (confirmation : Bool), 
    (confirmation = true) ∧ 
    (∃ (response : String), response = "Confirmed")

/-- Theorem: An SMS is genuine if and only if it meets the specified criteria -/
theorem sms_genuine_iff_criteria (s : SMS) (b : Bank) :
  is_genuine_sms s b ↔ 
  (s.sender = b.officialSMSNumber ∧ 
   ∃ (confirmation : Bool), 
     (confirmation = true) ∧ 
     (∃ (response : String), response = "Confirmed")) :=
by sorry

/-- Theorem: An SMS is a scam if and only if it doesn't meet the criteria for being genuine -/
theorem sms_scam_iff_not_genuine (s : SMS) (b : Bank) :
  ¬(is_genuine_sms s b) ↔ 
  (s.sender ≠ b.officialSMSNumber ∨ 
   ∀ (confirmation : Bool), 
     (confirmation = false) ∨ 
     (∀ (response : String), response ≠ "Confirmed")) :=
by sorry

end NUMINAMATH_CALUDE_sms_genuine_iff_criteria_sms_scam_iff_not_genuine_l2807_280728


namespace NUMINAMATH_CALUDE_regular_pentagon_ratio_sum_l2807_280762

/-- For a regular pentagon with side length a and diagonal length b, (a/b + b/a) = √5 -/
theorem regular_pentagon_ratio_sum (a b : ℝ) (h : a / b = (Real.sqrt 5 - 1) / 2) :
  a / b + b / a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_ratio_sum_l2807_280762


namespace NUMINAMATH_CALUDE_complex_square_simplification_l2807_280703

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 7 - 24 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l2807_280703


namespace NUMINAMATH_CALUDE_monkey_count_l2807_280757

theorem monkey_count : ∃! x : ℕ, x > 0 ∧ (x / 8)^2 + 12 = x := by
  sorry

end NUMINAMATH_CALUDE_monkey_count_l2807_280757
