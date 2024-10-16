import Mathlib

namespace NUMINAMATH_CALUDE_work_completion_time_l1878_187864

theorem work_completion_time 
  (original_men : ℕ) 
  (original_days : ℕ) 
  (absent_men : ℕ) 
  (final_days : ℕ) :
  original_men = 15 →
  original_days = 40 →
  absent_men = 5 →
  final_days = 60 →
  original_men * original_days = (original_men - absent_men) * final_days :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1878_187864


namespace NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l1878_187865

/-- Given two vectors a and b in ℝ², where a = (2,1) and b = (1-y, 2+y),
    and a is perpendicular to b, prove that |a - b| = 5√2. -/
theorem perpendicular_vectors_difference_magnitude :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ → ℝ × ℝ := λ y ↦ (1 - y, 2 + y)
  ∃ y : ℝ, (a.1 * (b y).1 + a.2 * (b y).2 = 0) →
    Real.sqrt ((a.1 - (b y).1)^2 + (a.2 - (b y).2)^2) = 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l1878_187865


namespace NUMINAMATH_CALUDE_manuels_savings_l1878_187854

/-- Calculates the total savings amount given initial deposit, weekly savings, and number of weeks -/
def totalSavings (initialDeposit weeklyAmount numWeeks : ℕ) : ℕ :=
  initialDeposit + weeklyAmount * numWeeks

/-- Theorem: Manuel's total savings after 19 weeks is $500 -/
theorem manuels_savings :
  totalSavings 177 17 19 = 500 := by
  sorry

end NUMINAMATH_CALUDE_manuels_savings_l1878_187854


namespace NUMINAMATH_CALUDE_meixian_kiwi_profit_1200_meixian_kiwi_profit_1800_impossible_l1878_187836

/-- Represents the kiwi sale scenario -/
structure KiwiSale where
  purchase_price : ℝ
  initial_selling_price : ℝ
  initial_sales : ℝ
  sales_increase_rate : ℝ

/-- Calculates the daily profit for a given price reduction -/
def daily_profit (ks : KiwiSale) (price_reduction : ℝ) : ℝ :=
  (ks.initial_selling_price - price_reduction - ks.purchase_price) *
  (ks.initial_sales + ks.sales_increase_rate * price_reduction)

/-- The kiwi sale scenario from the problem -/
def meixian_kiwi_sale : KiwiSale :=
  { purchase_price := 80
    initial_selling_price := 120
    initial_sales := 20
    sales_increase_rate := 2 }

theorem meixian_kiwi_profit_1200 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  daily_profit meixian_kiwi_sale x₁ = 1200 ∧
  daily_profit meixian_kiwi_sale x₂ = 1200 ∧
  (x₁ = 10 ∨ x₁ = 20) ∧ (x₂ = 10 ∨ x₂ = 20) :=
sorry

theorem meixian_kiwi_profit_1800_impossible :
  ¬∃ y : ℝ, daily_profit meixian_kiwi_sale y = 1800 :=
sorry

end NUMINAMATH_CALUDE_meixian_kiwi_profit_1200_meixian_kiwi_profit_1800_impossible_l1878_187836


namespace NUMINAMATH_CALUDE_root_sum_squares_reciprocal_l1878_187840

theorem root_sum_squares_reciprocal (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 49/36 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_reciprocal_l1878_187840


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1878_187857

/-- An isosceles triangle with side lengths 2, 2, and 5 has a perimeter of 9 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b c : ℝ),
      a = 2 ∧ b = 2 ∧ c = 5 ∧  -- Two sides are 2, one side is 5
      (a = b ∨ b = c ∨ a = c) ∧  -- Triangle is isosceles
      a + b > c ∧ b + c > a ∧ a + c > b ∧  -- Triangle inequality
      perimeter = a + b + c ∧  -- Definition of perimeter
      perimeter = 9  -- The perimeter is 9

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 9 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1878_187857


namespace NUMINAMATH_CALUDE_problem_solution_l1878_187800

noncomputable def f (a : ℝ) (θ : ℝ) (x : ℝ) : ℝ :=
  (a + 2 * (Real.cos x)^2) * Real.cos (2 * x + θ)

theorem problem_solution (a θ α : ℝ) :
  (∀ x, f a θ x = -f a θ (-x)) →  -- f is an odd function
  f a θ (π/4) = 0 →
  θ ∈ Set.Ioo 0 π →
  f a θ (α/4) = -2/5 →
  α ∈ Set.Ioo (π/2) π →
  (a = -1 ∧ θ = π/2 ∧ Real.sin (α + π/3) = (4 - 3 * Real.sqrt 3) / 10) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1878_187800


namespace NUMINAMATH_CALUDE_circle_arrangement_exists_l1878_187855

theorem circle_arrangement_exists : ∃ (a : Fin 12 → Fin 12), Function.Bijective a ∧
  ∀ (i j : Fin 12), i < j → |a i - a j| ≠ |i - j| := by
  sorry

end NUMINAMATH_CALUDE_circle_arrangement_exists_l1878_187855


namespace NUMINAMATH_CALUDE_binomial_expansion_97_cubed_l1878_187877

theorem binomial_expansion_97_cubed : 97^3 + 3*(97^2) + 3*97 + 1 = 941192 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_97_cubed_l1878_187877


namespace NUMINAMATH_CALUDE_negative_square_range_l1878_187849

theorem negative_square_range (x : ℝ) (h : -1 < x ∧ x < 0) : -1 < -x^2 ∧ -x^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_range_l1878_187849


namespace NUMINAMATH_CALUDE_matrix_power_2023_l1878_187889

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 0, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 2023; 0, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l1878_187889


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1878_187830

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, a * x + y - 1 - a = 0 ↔ x - (1/2) * y = 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l1878_187830


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l1878_187892

theorem increasing_function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_sum_positive : a + b > 0) : 
  f a + f b > f (-a) + f (-b) := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l1878_187892


namespace NUMINAMATH_CALUDE_other_piece_price_is_96_l1878_187846

/-- The price of one of the other pieces of clothing --/
def other_piece_price (total_spent : ℕ) (num_pieces : ℕ) (price1 : ℕ) (price2 : ℕ) : ℕ :=
  (total_spent - price1 - price2) / (num_pieces - 2)

/-- Theorem stating that the price of one of the other pieces is 96 --/
theorem other_piece_price_is_96 :
  other_piece_price 610 7 49 81 = 96 := by
  sorry

end NUMINAMATH_CALUDE_other_piece_price_is_96_l1878_187846


namespace NUMINAMATH_CALUDE_alberts_remaining_laps_l1878_187858

/-- Calculates the remaining laps for Albert's run -/
theorem alberts_remaining_laps 
  (total_distance : ℕ) 
  (track_length : ℕ) 
  (laps_run : ℕ) 
  (h1 : total_distance = 99) 
  (h2 : track_length = 9) 
  (h3 : laps_run = 6) : 
  total_distance / track_length - laps_run = 5 := by
  sorry

#check alberts_remaining_laps

end NUMINAMATH_CALUDE_alberts_remaining_laps_l1878_187858


namespace NUMINAMATH_CALUDE_triangle_count_48_l1878_187880

/-- The number of distinct, non-degenerate triangles with integer side lengths and perimeter n -/
def count_triangles (n : ℕ) : ℕ :=
  let isosceles := (n - 1) / 2 - n / 4
  let scalene := Nat.choose (n - 1) 2 - 3 * Nat.choose (n / 2) 2
  let total := (scalene - 3 * isosceles) / 6
  if n % 3 = 0 then total - 1 else total

theorem triangle_count_48 :
  ∃ n : ℕ, n > 0 ∧ count_triangles n = 48 :=
sorry

end NUMINAMATH_CALUDE_triangle_count_48_l1878_187880


namespace NUMINAMATH_CALUDE_all_subjects_identified_l1878_187804

theorem all_subjects_identified (num_colors : ℕ) (num_subjects : ℕ) : 
  num_colors = 5 → num_subjects = 16 → num_colors ^ 2 ≥ num_subjects := by
  sorry

#check all_subjects_identified

end NUMINAMATH_CALUDE_all_subjects_identified_l1878_187804


namespace NUMINAMATH_CALUDE_total_coffee_consumption_l1878_187844

/-- The number of cups of coffee Brayan drinks in an hour -/
def brayan_hourly_consumption : ℕ := 4

/-- The number of cups of coffee Ivory drinks in an hour -/
def ivory_hourly_consumption : ℕ := brayan_hourly_consumption / 2

/-- The total number of hours they drink coffee -/
def total_hours : ℕ := 5

/-- Theorem stating the total number of cups of coffee Ivory and Brayan drink together in 5 hours -/
theorem total_coffee_consumption :
  (brayan_hourly_consumption + ivory_hourly_consumption) * total_hours = 30 :=
by sorry

end NUMINAMATH_CALUDE_total_coffee_consumption_l1878_187844


namespace NUMINAMATH_CALUDE_interest_group_members_l1878_187816

/-- Represents a math interest group -/
structure InterestGroup where
  members : ℕ
  average_age : ℝ

/-- The change in average age when members leave or join -/
def age_change (g : InterestGroup) : Prop :=
  (g.members * g.average_age - 5 * 9 = (g.average_age + 1) * (g.members - 5)) ∧
  (g.members * g.average_age + 17 * 5 = (g.average_age + 1) * (g.members + 5))

theorem interest_group_members :
  ∃ (g : InterestGroup), age_change g → g.members = 20 := by
  sorry

end NUMINAMATH_CALUDE_interest_group_members_l1878_187816


namespace NUMINAMATH_CALUDE_alcohol_concentration_in_mixture_l1878_187893

/-- Calculates the new concentration of alcohol in a mixture --/
theorem alcohol_concentration_in_mixture
  (vessel1_capacity : ℝ)
  (vessel1_concentration : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_concentration : ℝ)
  (total_liquid : ℝ)
  (new_vessel_capacity : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_concentration = 0.4)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_concentration = 0.6)
  (h5 : total_liquid = 8)
  (h6 : new_vessel_capacity = 10)
  (h7 : total_liquid ≤ new_vessel_capacity) :
  let alcohol1 := vessel1_capacity * vessel1_concentration
  let alcohol2 := vessel2_capacity * vessel2_concentration
  let total_alcohol := alcohol1 + alcohol2
  let water_added := new_vessel_capacity - total_liquid
  let new_concentration := total_alcohol / new_vessel_capacity
  new_concentration = 0.44 := by
  sorry

#check alcohol_concentration_in_mixture

end NUMINAMATH_CALUDE_alcohol_concentration_in_mixture_l1878_187893


namespace NUMINAMATH_CALUDE_kolya_can_break_rods_to_form_triangles_l1878_187823

/-- Represents a rod broken into three parts -/
structure BrokenRod :=
  (part1 : ℝ)
  (part2 : ℝ)
  (part3 : ℝ)
  (sum_to_one : part1 + part2 + part3 = 1)
  (all_positive : part1 > 0 ∧ part2 > 0 ∧ part3 > 0)

/-- Checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Checks if it's possible to form three triangles from three broken rods -/
def can_form_three_triangles (rod1 rod2 rod3 : BrokenRod) : Prop :=
  ∃ (perm1 perm2 perm3 : Fin 3 → Fin 3),
    can_form_triangle (rod1.part1) (rod2.part1) (rod3.part1) ∧
    can_form_triangle (rod1.part2) (rod2.part2) (rod3.part2) ∧
    can_form_triangle (rod1.part3) (rod2.part3) (rod3.part3)

/-- The main theorem stating that Kolya can break the rods to always form three triangles -/
theorem kolya_can_break_rods_to_form_triangles :
  ∃ (kolya_rod1 kolya_rod2 : BrokenRod),
    ∀ (vasya_rod : BrokenRod),
      can_form_three_triangles kolya_rod1 vasya_rod kolya_rod2 :=
sorry

end NUMINAMATH_CALUDE_kolya_can_break_rods_to_form_triangles_l1878_187823


namespace NUMINAMATH_CALUDE_complex_power_modulus_l1878_187819

theorem complex_power_modulus : Complex.abs ((5 : ℂ) + (2 * Complex.I * Real.sqrt 3))^4 = 1369 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l1878_187819


namespace NUMINAMATH_CALUDE_difference_of_squares_l1878_187821

theorem difference_of_squares (x : ℝ) : x^2 - 121 = (x + 11) * (x - 11) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1878_187821


namespace NUMINAMATH_CALUDE_percentage_problem_l1878_187810

theorem percentage_problem (x : ℝ) (h : 0.3 * x = 120) : 0.4 * x = 160 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1878_187810


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_48_l1878_187885

theorem smallest_positive_multiple_of_48 :
  ∀ n : ℕ, n > 0 → 48 ∣ n → n ≥ 48 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_48_l1878_187885


namespace NUMINAMATH_CALUDE_kindergarten_distribution_l1878_187882

def apples : ℕ := 270
def pears : ℕ := 180
def oranges : ℕ := 235

def is_valid_distribution (n : ℕ) : Prop :=
  n ≠ 0 ∧
  (apples - n * (apples / n) : ℤ) = 3 * (oranges - n * (oranges / n)) ∧
  (pears - n * (pears / n) : ℤ) = 2 * (oranges - n * (oranges / n))

theorem kindergarten_distribution :
  ∃ (n : ℕ), is_valid_distribution n ∧ n = 29 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_distribution_l1878_187882


namespace NUMINAMATH_CALUDE_lower_parallel_length_l1878_187812

/-- A triangle with a base of 20 inches and two parallel lines dividing it into four equal areas -/
structure EqualAreaTriangle where
  /-- The base of the triangle -/
  base : ℝ
  /-- The length of the parallel line closer to the base -/
  lower_parallel : ℝ
  /-- The base is 20 inches -/
  base_length : base = 20
  /-- The parallel lines divide the triangle into four equal areas -/
  equal_areas : lower_parallel^2 / base^2 = 1/4

/-- The length of the parallel line closer to the base is 10 inches -/
theorem lower_parallel_length (t : EqualAreaTriangle) : t.lower_parallel = 10 := by
  sorry

end NUMINAMATH_CALUDE_lower_parallel_length_l1878_187812


namespace NUMINAMATH_CALUDE_cubic_root_values_l1878_187867

theorem cubic_root_values (m : ℝ) : 
  (-1 : ℂ)^3 - (m^2 - m + 7)*(-1 : ℂ) - (3*m^2 - 3*m - 6) = 0 ↔ m = -2 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_values_l1878_187867


namespace NUMINAMATH_CALUDE_division_22_by_8_l1878_187852

theorem division_22_by_8 : (22 : ℚ) / 8 = 2.75 := by sorry

end NUMINAMATH_CALUDE_division_22_by_8_l1878_187852


namespace NUMINAMATH_CALUDE_total_spent_is_12_30_l1878_187824

/-- The cost of the football Alyssa bought -/
def football_cost : ℚ := 571/100

/-- The cost of the marbles Alyssa bought -/
def marbles_cost : ℚ := 659/100

/-- The total amount Alyssa spent on toys -/
def total_spent : ℚ := football_cost + marbles_cost

/-- Theorem stating that the total amount Alyssa spent on toys is $12.30 -/
theorem total_spent_is_12_30 : total_spent = 1230/100 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_12_30_l1878_187824


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1878_187828

theorem modulus_of_complex_fraction (z : ℂ) : z = (1 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1878_187828


namespace NUMINAMATH_CALUDE_geometric_series_sum_quarter_five_terms_l1878_187856

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_quarter_five_terms :
  geometric_series_sum (1/4) (1/4) 5 = 341/1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_quarter_five_terms_l1878_187856


namespace NUMINAMATH_CALUDE_books_count_l1878_187829

/-- The number of books Darryl has -/
def darryl_books : ℕ := 20

/-- The number of books Lamont has -/
def lamont_books : ℕ := 2 * darryl_books

/-- The number of books Loris has -/
def loris_books : ℕ := lamont_books - 3

/-- The total number of books all three have -/
def total_books : ℕ := darryl_books + lamont_books + loris_books

theorem books_count : total_books = 97 := by
  sorry

end NUMINAMATH_CALUDE_books_count_l1878_187829


namespace NUMINAMATH_CALUDE_intersection_complement_equals_interval_l1878_187809

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 < 1}
def B : Set ℝ := {x : ℝ | x^2 - 2*x > 0}

-- State the theorem
theorem intersection_complement_equals_interval :
  A ∩ (Set.univ \ B) = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_interval_l1878_187809


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l1878_187869

open Set

theorem fixed_point_theorem (f g : (Set.Icc 0 1) → (Set.Icc 0 1))
  (hf_cont : Continuous f)
  (hg_cont : Continuous g)
  (h_comm : ∀ x ∈ Set.Icc 0 1, f (g x) = g (f x))
  (hf_incr : StrictMono f) :
  ∃ a ∈ Set.Icc 0 1, f a = a ∧ g a = a := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l1878_187869


namespace NUMINAMATH_CALUDE_consecutive_integers_product_552_l1878_187850

theorem consecutive_integers_product_552 (x : ℕ) 
  (h1 : x > 0) 
  (h2 : x * (x + 1) = 552) : 
  x + (x + 1) = 47 ∧ (x + 1) - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_552_l1878_187850


namespace NUMINAMATH_CALUDE_f_min_max_l1878_187871

noncomputable def f (x : ℝ) : ℝ := 3 - 4 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4

theorem f_min_max : ∃ (m M : ℝ), (∀ x, m ≤ f x ∧ f x ≤ M) ∧ (∃ x₁ x₂, f x₁ = m ∧ f x₂ = M) ∧ m = 2 ∧ M = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_min_max_l1878_187871


namespace NUMINAMATH_CALUDE_altitude_length_l1878_187847

/-- Given a rectangle with length l and width w, and a triangle constructed on its diagonal
    with an area equal to the rectangle's area, the length of the altitude drawn from the
    opposite vertex of the triangle to the diagonal is (2lw) / √(l^2 + w^2). -/
theorem altitude_length (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let diagonal := Real.sqrt (l^2 + w^2)
  let rectangle_area := l * w
  let triangle_area := (1/2) * diagonal * altitude
  altitude = (2 * l * w) / diagonal →
  triangle_area = rectangle_area :=
by
  sorry


end NUMINAMATH_CALUDE_altitude_length_l1878_187847


namespace NUMINAMATH_CALUDE_arithmetic_sequence_triangle_cosine_identity_l1878_187875

theorem arithmetic_sequence_triangle_cosine_identity (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  2 * b = a + c →
  5 * Real.cos A - 4 * Real.cos A * Real.cos C + 5 * Real.cos C = 8 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_triangle_cosine_identity_l1878_187875


namespace NUMINAMATH_CALUDE_y_coordinates_equal_l1878_187805

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := y = 2 * (x - 3)^2 - 4

theorem y_coordinates_equal :
  ∀ y₁ y₂ : ℝ,
  parabola 2 y₁ →
  parabola 4 y₂ →
  y₁ = y₂ := by
  sorry

end NUMINAMATH_CALUDE_y_coordinates_equal_l1878_187805


namespace NUMINAMATH_CALUDE_polynomial_evaluation_part1_polynomial_evaluation_part2_l1878_187860

/-- Part 1: Polynomial evaluation given a condition -/
theorem polynomial_evaluation_part1 (x : ℝ) (h : x^2 - x = 3) :
  x^4 - 2*x^3 + 3*x^2 - 2*x + 2 = 17 := by
  sorry

/-- Part 2: Polynomial evaluation given a condition -/
theorem polynomial_evaluation_part2 (x y : ℝ) (h : x^2 + y^2 = 1) :
  2*x^4 + 3*x^2*y^2 + y^4 + y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_part1_polynomial_evaluation_part2_l1878_187860


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1878_187898

theorem sqrt_x_minus_one_real (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1878_187898


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1878_187839

theorem quadratic_equation_solution (x m k : ℝ) :
  (x + m) * (x - 5) = x^2 - 3*x + k →
  k = -10 ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1878_187839


namespace NUMINAMATH_CALUDE_adjacent_zero_point_functions_range_l1878_187861

def adjacent_zero_point_functions (f g : ℝ → ℝ) : Prop :=
  ∀ (α β : ℝ), f α = 0 → g β = 0 → |α - β| ≤ 1

def f (x : ℝ) : ℝ := x - 1

def g (a x : ℝ) : ℝ := x^2 - a*x - a + 3

theorem adjacent_zero_point_functions_range (a : ℝ) :
  adjacent_zero_point_functions f (g a) → a ∈ Set.Icc 2 (7/3) := by
  sorry

end NUMINAMATH_CALUDE_adjacent_zero_point_functions_range_l1878_187861


namespace NUMINAMATH_CALUDE_max_abs_value_l1878_187818

theorem max_abs_value (x y : ℝ) 
  (h1 : x + y - 2 ≤ 0) 
  (h2 : x - y + 4 ≥ 0) 
  (h3 : y ≥ 0) : 
  ∃ (z : ℝ), z = |x - 2*y + 2| ∧ z ≤ 5 ∧ ∀ (w : ℝ), w = |x - 2*y + 2| → w ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_abs_value_l1878_187818


namespace NUMINAMATH_CALUDE_calculate_expression_l1878_187831

theorem calculate_expression : 75 * 1313 - 25 * 1313 = 65750 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1878_187831


namespace NUMINAMATH_CALUDE_player5_score_breakdown_l1878_187841

/-- Represents the scoring breakdown for a basketball player -/
structure PlayerScore where
  threes : Nat
  twos : Nat
  frees : Nat

/-- Calculates the total points scored by a player -/
def totalPoints (score : PlayerScore) : Nat :=
  3 * score.threes + 2 * score.twos + score.frees

theorem player5_score_breakdown :
  ∀ (team_total : Nat) (other_players_total : Nat),
    team_total = 75 →
    other_players_total = 61 →
    ∃ (score : PlayerScore),
      totalPoints score = team_total - other_players_total ∧
      score.threes ≥ 2 ∧
      score.twos ≥ 1 ∧
      score.frees ≤ 4 ∧
      score.threes = 2 ∧
      score.twos = 2 ∧
      score.frees = 4 :=
by sorry

end NUMINAMATH_CALUDE_player5_score_breakdown_l1878_187841


namespace NUMINAMATH_CALUDE_election_majority_l1878_187820

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 500 →
  winning_percentage = 70/100 →
  (winning_percentage * total_votes : ℚ).num - ((1 - winning_percentage) * total_votes : ℚ).num = 200 := by
sorry

end NUMINAMATH_CALUDE_election_majority_l1878_187820


namespace NUMINAMATH_CALUDE_probability_x_greater_than_3y_l1878_187894

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  y_min : ℝ
  x_max : ℝ
  y_max : ℝ

/-- A point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is inside a rectangle --/
def Point.insideRectangle (p : Point) (r : Rectangle) : Prop :=
  r.x_min ≤ p.x ∧ p.x ≤ r.x_max ∧ r.y_min ≤ p.y ∧ p.y ≤ r.y_max

/-- The probability of an event occurring for a point randomly picked from a rectangle --/
def probability (r : Rectangle) (event : Point → Prop) : ℝ :=
  sorry

/-- The specific rectangle in the problem --/
def problemRectangle : Rectangle :=
  { x_min := 0, y_min := 0, x_max := 3000, y_max := 3000 }

/-- The event x > 3y --/
def xGreaterThan3y (p : Point) : Prop :=
  p.x > 3 * p.y

theorem probability_x_greater_than_3y :
  probability problemRectangle xGreaterThan3y = 1/6 :=
sorry

end NUMINAMATH_CALUDE_probability_x_greater_than_3y_l1878_187894


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l1878_187803

/-- Conversion from polar coordinates to rectangular coordinates --/
theorem polar_to_rectangular (r θ : ℝ) :
  r = 6 ∧ θ = π / 3 →
  ∃ x y : ℝ, x = 3 ∧ y = 3 * Real.sqrt 3 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l1878_187803


namespace NUMINAMATH_CALUDE_giant_slide_rides_count_l1878_187813

/-- Represents the carnival scenario with given ride times and planned rides --/
structure CarnivalScenario where
  total_time : ℕ  -- Total time in minutes
  roller_coaster_time : ℕ
  tilt_a_whirl_time : ℕ
  giant_slide_time : ℕ
  vortex_time : ℕ
  bumper_cars_time : ℕ
  roller_coaster_rides : ℕ
  tilt_a_whirl_rides : ℕ
  vortex_rides : ℕ
  bumper_cars_rides : ℕ

/-- Theorem stating that the number of giant slide rides is equal to tilt-a-whirl rides --/
theorem giant_slide_rides_count (scenario : CarnivalScenario) : 
  scenario.total_time = 240 ∧
  scenario.roller_coaster_time = 30 ∧
  scenario.tilt_a_whirl_time = 60 ∧
  scenario.giant_slide_time = 15 ∧
  scenario.vortex_time = 45 ∧
  scenario.bumper_cars_time = 25 ∧
  scenario.roller_coaster_rides = 4 ∧
  scenario.tilt_a_whirl_rides = 2 ∧
  scenario.vortex_rides = 1 ∧
  scenario.bumper_cars_rides = 3 →
  scenario.tilt_a_whirl_rides = 2 :=
by sorry

end NUMINAMATH_CALUDE_giant_slide_rides_count_l1878_187813


namespace NUMINAMATH_CALUDE_solution_set_implies_m_value_l1878_187853

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 3|

-- State the theorem
theorem solution_set_implies_m_value (m : ℝ) :
  (∀ x : ℝ, f m x > 2 ↔ 2 < x ∧ x < 4) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_value_l1878_187853


namespace NUMINAMATH_CALUDE_max_z_value_l1878_187883

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 0) (h2 : x*y + y*z + z*x = -3) :
  ∃ (max_z : ℝ), z ≤ max_z ∧ max_z = 2 := by
sorry

end NUMINAMATH_CALUDE_max_z_value_l1878_187883


namespace NUMINAMATH_CALUDE_shark_sightings_relationship_l1878_187863

/-- The number of shark sightings in Daytona Beach per year. -/
def daytona_sightings : ℕ := 26

/-- The number of shark sightings in Cape May per year. -/
def cape_may_sightings : ℕ := 7

/-- Theorem stating the relationship between shark sightings in Daytona Beach and Cape May. -/
theorem shark_sightings_relationship :
  daytona_sightings = 3 * cape_may_sightings + 5 ∧ cape_may_sightings = 7 := by
  sorry

end NUMINAMATH_CALUDE_shark_sightings_relationship_l1878_187863


namespace NUMINAMATH_CALUDE_M_intersect_N_is_empty_l1878_187876

-- Define set M
def M : Set ℝ := {y | ∃ x, y = x + 1}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

-- Theorem statement
theorem M_intersect_N_is_empty : M ∩ (N.image Prod.snd) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_is_empty_l1878_187876


namespace NUMINAMATH_CALUDE_star_three_four_l1878_187868

def star (x y : ℝ) : ℝ := 4 * x + 6 * y

theorem star_three_four : star 3 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_star_three_four_l1878_187868


namespace NUMINAMATH_CALUDE_root_in_interval_l1878_187825

-- Define the function f(x) = x³ - x - 3
def f (x : ℝ) : ℝ := x^3 - x - 3

-- State the theorem
theorem root_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 1 2 ∧ f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l1878_187825


namespace NUMINAMATH_CALUDE_koala_fiber_intake_l1878_187866

/-- The absorption rate of fiber for koalas -/
def absorption_rate : ℝ := 0.35

/-- The amount of fiber absorbed on the first day -/
def fiber_absorbed_day1 : ℝ := 14.7

/-- The amount of fiber absorbed on the second day -/
def fiber_absorbed_day2 : ℝ := 9.8

/-- Theorem: The total amount of fiber eaten by the koala over two days is 70 ounces -/
theorem koala_fiber_intake :
  let fiber_eaten_day1 := fiber_absorbed_day1 / absorption_rate
  let fiber_eaten_day2 := fiber_absorbed_day2 / absorption_rate
  fiber_eaten_day1 + fiber_eaten_day2 = 70 := by sorry

end NUMINAMATH_CALUDE_koala_fiber_intake_l1878_187866


namespace NUMINAMATH_CALUDE_jar_price_proportion_l1878_187886

/-- Given two cylindrical jars with diameters d₁ and d₂, heights h₁ and h₂, 
    and the price of the first jar p₁, if the price is proportional to the volume, 
    then the price of the second jar p₂ is equal to p₁ * (d₂/d₁)² * (h₂/h₁). -/
theorem jar_price_proportion (d₁ d₂ h₁ h₂ p₁ p₂ : ℝ) (h_d₁_pos : d₁ > 0) (h_d₂_pos : d₂ > 0) 
    (h_h₁_pos : h₁ > 0) (h_h₂_pos : h₂ > 0) (h_p₁_pos : p₁ > 0) :
  p₂ = p₁ * (d₂/d₁)^2 * (h₂/h₁) ↔ 
  p₂ / (π * (d₂/2)^2 * h₂) = p₁ / (π * (d₁/2)^2 * h₁) := by
sorry

end NUMINAMATH_CALUDE_jar_price_proportion_l1878_187886


namespace NUMINAMATH_CALUDE_fraction_difference_equals_sqrt_five_l1878_187815

theorem fraction_difference_equals_sqrt_five (a b : ℝ) (h1 : a ≠ b) (h2 : 1/a + 1/b = Real.sqrt 5) :
  a / (b * (a - b)) - b / (a * (a - b)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_sqrt_five_l1878_187815


namespace NUMINAMATH_CALUDE_buckingham_palace_visitors_l1878_187801

/-- The difference in visitors between the current day and the sum of the previous two days -/
def visitor_difference (current_day : ℕ) (previous_day : ℕ) (two_days_ago : ℕ) : ℤ :=
  (current_day : ℤ) - (previous_day + two_days_ago : ℤ)

/-- Theorem stating the visitor difference for the given numbers -/
theorem buckingham_palace_visitors :
  visitor_difference 1321 890 765 = -334 := by
  sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitors_l1878_187801


namespace NUMINAMATH_CALUDE_room_tiles_count_l1878_187833

/-- Calculates the least number of square tiles required to cover a rectangular floor. -/
def leastSquareTiles (length width : ℕ) : ℕ :=
  let tileSize := Nat.gcd length width
  (length / tileSize) * (width / tileSize)

/-- Theorem stating that for a room with given dimensions, 153 square tiles are required. -/
theorem room_tiles_count :
  leastSquareTiles 816 432 = 153 := by
  sorry

#eval leastSquareTiles 816 432

end NUMINAMATH_CALUDE_room_tiles_count_l1878_187833


namespace NUMINAMATH_CALUDE_largest_angle_of_triangle_l1878_187807

/-- Given a triangle XYZ with sides x, y, and z satisfying certain conditions,
    prove that its largest angle is 120°. -/
theorem largest_angle_of_triangle (x y z : ℝ) (h1 : x + 3*y + 4*z = x^2) (h2 : x + 3*y - 4*z = -7) :
  ∃ (X Y Z : ℝ), X + Y + Z = 180 ∧ 0 < X ∧ 0 < Y ∧ 0 < Z ∧ max X (max Y Z) = 120 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_of_triangle_l1878_187807


namespace NUMINAMATH_CALUDE_opposite_sides_condition_l1878_187884

/-- 
Given a real number m, if the points (1, 2) and (1, 1) are on opposite sides of the line y - 3x - m = 0, 
then -2 < m < -1.
-/
theorem opposite_sides_condition (m : ℝ) : 
  (2 - 3 * 1 - m) * (1 - 3 * 1 - m) < 0 → -2 < m ∧ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_condition_l1878_187884


namespace NUMINAMATH_CALUDE_max_revenue_l1878_187881

/-- Represents the production quantities of products A and B -/
structure Production where
  a : ℝ
  b : ℝ

/-- Calculates the revenue for a given production -/
def revenue (p : Production) : ℝ :=
  0.3 * p.a + 0.2 * p.b

/-- Checks if a production is feasible given the machine constraints -/
def is_feasible (p : Production) : Prop :=
  p.a ≥ 0 ∧ p.b ≥ 0 ∧
  1 * p.a + 2 * p.b ≤ 400 ∧
  2 * p.a + 1 * p.b ≤ 500

/-- Theorem stating the maximum monthly sales revenue -/
theorem max_revenue :
  ∃ (p : Production), is_feasible p ∧
    ∀ (q : Production), is_feasible q → revenue q ≤ revenue p ∧
    revenue p = 90 :=
sorry

end NUMINAMATH_CALUDE_max_revenue_l1878_187881


namespace NUMINAMATH_CALUDE_existence_of_prime_not_divisible_l1878_187806

theorem existence_of_prime_not_divisible (p : Nat) (h_prime : Prime p) (h_p_gt_2 : p > 2) :
  ∃ q : Nat, Prime q ∧ q < p ∧ ¬(p^2 ∣ q^(p-1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_prime_not_divisible_l1878_187806


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l1878_187822

theorem maintenance_check_increase (original_time : ℝ) (percentage_increase : ℝ) 
  (h1 : original_time = 45)
  (h2 : percentage_increase = 33.33333333333333) : 
  original_time * (1 + percentage_increase / 100) = 60 := by
sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l1878_187822


namespace NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l1878_187887

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (m : ℝ) : Prop :=
  (1 : ℝ) / (-(1 + m)) = m / (-2 : ℝ)

/-- If the lines x + (1+m)y = 2-m and mx + 2y + 8 = 0 are parallel, then m = 1 -/
theorem parallel_lines_m_equals_one :
  ∀ m : ℝ, parallel_lines m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l1878_187887


namespace NUMINAMATH_CALUDE_right_triangle_sets_l1878_187834

theorem right_triangle_sets : 
  (¬ (4^2 + 6^2 = 8^2)) ∧ 
  (5^2 + 12^2 = 13^2) ∧ 
  (6^2 + 8^2 = 10^2) ∧ 
  (7^2 + 24^2 = 25^2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l1878_187834


namespace NUMINAMATH_CALUDE_simplify_fraction_l1878_187843

theorem simplify_fraction : (150 : ℚ) / 4350 = 1 / 29 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1878_187843


namespace NUMINAMATH_CALUDE_finance_equation_solution_l1878_187895

/-- Given the equation fp - w = 20000, where f = 4 and w = 10 + 200i, prove that p = 5002.5 + 50i. -/
theorem finance_equation_solution (f w p : ℂ) : 
  f = 4 → w = 10 + 200 * Complex.I → f * p - w = 20000 → p = 5002.5 + 50 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_finance_equation_solution_l1878_187895


namespace NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l1878_187874

def f (x : ℝ) : ℝ := x^2 + x

theorem f_increasing_on_positive_reals :
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l1878_187874


namespace NUMINAMATH_CALUDE_unique_base_number_for_16_factorial_l1878_187896

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem unique_base_number_for_16_factorial :
  ∃! b : ℕ+, b > 1 ∧ (factorial 16 % (b : ℕ)^6 = 0) ∧ (factorial 16 % (b : ℕ)^7 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_base_number_for_16_factorial_l1878_187896


namespace NUMINAMATH_CALUDE_power_of_power_l1878_187832

theorem power_of_power (a : ℝ) : (a^3)^4 = a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1878_187832


namespace NUMINAMATH_CALUDE_bank_teller_coins_l1878_187899

theorem bank_teller_coins (rolls_per_teller : ℕ) (coins_per_roll : ℕ) (num_tellers : ℕ) :
  rolls_per_teller = 10 →
  coins_per_roll = 25 →
  num_tellers = 4 →
  rolls_per_teller * coins_per_roll * num_tellers = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_bank_teller_coins_l1878_187899


namespace NUMINAMATH_CALUDE_sum_bound_l1878_187826

theorem sum_bound (k a b c : ℝ) (h1 : k > 1) (h2 : a ≥ 0) (h3 : b ≥ 0) (h4 : c ≥ 0)
  (h5 : a ≤ k * c) (h6 : b ≤ k * c) (h7 : a * b ≤ c^2) :
  a + b ≤ (k + 1/k) * c := by
  sorry

end NUMINAMATH_CALUDE_sum_bound_l1878_187826


namespace NUMINAMATH_CALUDE_double_flip_is_rotation_l1878_187842

/-- Represents a 2D point -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  P : Point2D
  Q : Point2D
  R : Point2D

/-- Represents a rotation of a triangle -/
def rotate_triangle (t : EquilateralTriangle) (angle : ℝ) : EquilateralTriangle :=
  sorry

/-- Represents flipping a triangle around an edge -/
def flip_triangle (t : EquilateralTriangle) (edge : Fin 3) : EquilateralTriangle :=
  sorry

/-- Theorem: Two consecutive flips of an equilateral triangle result in a 120-degree rotation -/
theorem double_flip_is_rotation (t : EquilateralTriangle) :
  ∃ (edge1 edge2 : Fin 3), edge1 ≠ edge2 →
    flip_triangle (flip_triangle t edge1) edge2 = rotate_triangle t (2 * Real.pi / 3) :=
  sorry

end NUMINAMATH_CALUDE_double_flip_is_rotation_l1878_187842


namespace NUMINAMATH_CALUDE_rich_walking_distance_l1878_187848

def house_to_sidewalk : ℕ := 20
def sidewalk_to_road_end : ℕ := 200

def total_distance : ℕ :=
  let to_road_end := house_to_sidewalk + sidewalk_to_road_end
  let to_intersection := to_road_end + 2 * to_road_end
  let to_route_end := to_intersection + to_intersection / 2
  2 * to_route_end

theorem rich_walking_distance :
  total_distance = 1980 := by
  sorry

end NUMINAMATH_CALUDE_rich_walking_distance_l1878_187848


namespace NUMINAMATH_CALUDE_identity_function_satisfies_condition_l1878_187879

theorem identity_function_satisfies_condition :
  ∀ (f : ℕ+ → ℕ+),
    (∀ n : ℕ+, f n > f (f (n - 1))) →
    (∀ n : ℕ+, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_identity_function_satisfies_condition_l1878_187879


namespace NUMINAMATH_CALUDE_min_sum_of_quadratic_roots_l1878_187817

theorem min_sum_of_quadratic_roots (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ x : ℝ, x^2 + m*x + 2*n = 0) → 
  (∃ x : ℝ, x^2 + 2*n*x + m = 0) → 
  m + n ≥ 6 := by sorry

end NUMINAMATH_CALUDE_min_sum_of_quadratic_roots_l1878_187817


namespace NUMINAMATH_CALUDE_specific_pentagon_area_l1878_187845

/-- Pentagon formed by cutting a right-angled triangular corner from a rectangle -/
structure Pentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  parallel_sides : Bool
  triangle_leg1 : ℝ
  triangle_leg2 : ℝ

/-- The area of the pentagon -/
def pentagon_area (p : Pentagon) : ℝ := sorry

/-- Theorem stating the area of the specific pentagon -/
theorem specific_pentagon_area :
  ∃ (p : Pentagon),
    p.side1 = 9 ∧ p.side2 = 16 ∧ p.side3 = 30 ∧ p.side4 = 40 ∧ p.side5 = 41 ∧
    p.parallel_sides = true ∧
    p.triangle_leg1 = 9 ∧ p.triangle_leg2 = 40 ∧
    pentagon_area p = 1020 := by
  sorry

end NUMINAMATH_CALUDE_specific_pentagon_area_l1878_187845


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1878_187891

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    p ∣ (3^11 + 5^13) ∧ 
    ∀ (q : ℕ), Nat.Prime q → q ∣ (3^11 + 5^13) → p ≤ q ∧
    p = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1878_187891


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l1878_187890

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 4 ≥ 0) → a ∈ Set.Icc (-4 : ℝ) 4 := by
sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l1878_187890


namespace NUMINAMATH_CALUDE_product_even_implies_one_even_one_odd_l1878_187897

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- Checks if all coefficients of a polynomial are even -/
def allEven (p : IntPolynomial n) : Prop :=
  ∀ i, 2 ∣ p i

/-- Checks if at least one coefficient of a polynomial is odd -/
def hasOddCoeff (p : IntPolynomial n) : Prop :=
  ∃ i, ¬(2 ∣ p i)

/-- Represents the product of two polynomials -/
def polyProduct (a : IntPolynomial n) (b : IntPolynomial m) : IntPolynomial (n + m - 1) :=
  sorry

/-- Main theorem -/
theorem product_even_implies_one_even_one_odd
  (n m : ℕ)
  (a : IntPolynomial n)
  (b : IntPolynomial m)
  (h1 : allEven (polyProduct a b))
  (h2 : ∃ i, ¬(4 ∣ (polyProduct a b) i)) :
  (allEven a ∧ hasOddCoeff b) ∨ (allEven b ∧ hasOddCoeff a) :=
sorry

end NUMINAMATH_CALUDE_product_even_implies_one_even_one_odd_l1878_187897


namespace NUMINAMATH_CALUDE_largest_A_when_quotient_equals_remainder_l1878_187802

theorem largest_A_when_quotient_equals_remainder (A B C : ℕ) : 
  A = 7 * B + C → B = C → A ≤ 48 ∧ ∃ (A₀ B₀ C₀ : ℕ), A₀ = 7 * B₀ + C₀ ∧ B₀ = C₀ ∧ A₀ = 48 := by
  sorry

end NUMINAMATH_CALUDE_largest_A_when_quotient_equals_remainder_l1878_187802


namespace NUMINAMATH_CALUDE_rectangular_box_height_l1878_187811

theorem rectangular_box_height (wooden_box_length wooden_box_width wooden_box_height : ℕ)
  (box_length box_width : ℕ) (max_boxes : ℕ) :
  wooden_box_length = 800 ∧ wooden_box_width = 700 ∧ wooden_box_height = 600 ∧
  box_length = 8 ∧ box_width = 7 ∧ max_boxes = 1000000 →
  ∃ (box_height : ℕ), 
    (wooden_box_length * wooden_box_width * wooden_box_height) / max_boxes = 
    box_length * box_width * box_height ∧ box_height = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_height_l1878_187811


namespace NUMINAMATH_CALUDE_prime_square_mod_60_l1878_187827

-- Define the set of primes greater than 3
def PrimesGreaterThan3 : Set ℕ := {p : ℕ | Nat.Prime p ∧ p > 3}

-- Theorem statement
theorem prime_square_mod_60 (p : ℕ) (h : p ∈ PrimesGreaterThan3) : 
  p ^ 2 % 60 = 1 ∨ p ^ 2 % 60 = 49 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_60_l1878_187827


namespace NUMINAMATH_CALUDE_always_solution_never_solution_l1878_187862

-- Define the polynomial function
def f (a x : ℝ) : ℝ := (1 + a) * x^4 + x^3 - (3*a + 2) * x^2 - 4*a

-- Theorem 1: For all real a, x = -2 is a solution
theorem always_solution : ∀ a : ℝ, f a (-2) = 0 := by sorry

-- Theorem 2: For all real a, x = 2 is not a solution
theorem never_solution : ∀ a : ℝ, f a 2 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_always_solution_never_solution_l1878_187862


namespace NUMINAMATH_CALUDE_power_function_inequality_l1878_187859

-- Define the power function
def f (x : ℝ) : ℝ := x^(1/5)

-- State the theorem
theorem power_function_inequality (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) :
  f ((x1 + x2) / 2) > (f x1 + f x2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_inequality_l1878_187859


namespace NUMINAMATH_CALUDE_unique_k_for_triple_f_l1878_187878

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem unique_k_for_triple_f (k : ℤ) : 
  k % 2 = 1 ∧ f (f (f k)) = 57 → k = 223 :=
by sorry

end NUMINAMATH_CALUDE_unique_k_for_triple_f_l1878_187878


namespace NUMINAMATH_CALUDE_problem_solution_l1878_187872

theorem problem_solution (h : 43 * 47 = 2021) : (-43) / (1 / 47) = -2021 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1878_187872


namespace NUMINAMATH_CALUDE_tim_income_compared_to_juan_tim_percent_less_than_juan_l1878_187870

theorem tim_income_compared_to_juan (tim mary juan : ℝ) 
  (h1 : mary = 1.5 * tim) 
  (h2 : mary = 0.8999999999999999 * juan) : 
  tim = 0.6 * juan :=
sorry

theorem tim_percent_less_than_juan (tim juan : ℝ) 
  (h : tim = 0.6 * juan) : 
  (juan - tim) / juan = 0.4 :=
sorry

end NUMINAMATH_CALUDE_tim_income_compared_to_juan_tim_percent_less_than_juan_l1878_187870


namespace NUMINAMATH_CALUDE_lesser_number_problem_l1878_187808

theorem lesser_number_problem (x y : ℝ) (h1 : x + y = 50) (h2 : x * y = 612) :
  min x y = 21.395 := by sorry

end NUMINAMATH_CALUDE_lesser_number_problem_l1878_187808


namespace NUMINAMATH_CALUDE_xyz_product_l1878_187873

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -25)
  (eq2 : y * z + 5 * z = -25)
  (eq3 : z * x + 5 * x = -25) :
  x * y * z = 125 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l1878_187873


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1878_187838

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t ≠ 0 ∧ a.1 = t * b.1 ∧ a.2 = t * b.2

/-- The theorem stating that if (1,k) is parallel to (2,1), then k = 1/2 -/
theorem parallel_vectors_k_value (k : ℝ) :
  are_parallel (1, k) (2, 1) → k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1878_187838


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l1878_187814

theorem triangle_inequality_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + 2*b*c) / (b^2 + c^2) + (b^2 + 2*a*c) / (c^2 + a^2) + (c^2 + 2*a*b) / (a^2 + b^2) > 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l1878_187814


namespace NUMINAMATH_CALUDE_plant_species_numbering_impossibility_l1878_187835

theorem plant_species_numbering_impossibility :
  ∃ (a b : ℕ), 2 ≤ a ∧ a < b ∧ b ≤ 20000 ∧
  (∀ (x : ℕ), 2 ≤ x ∧ x ≤ 20000 →
    (Nat.gcd a x = Nat.gcd b x)) :=
sorry

end NUMINAMATH_CALUDE_plant_species_numbering_impossibility_l1878_187835


namespace NUMINAMATH_CALUDE_max_value_2ac_minus_abc_l1878_187837

theorem max_value_2ac_minus_abc : 
  ∀ a b c : ℕ+, 
  a ≤ 7 → b ≤ 6 → c ≤ 4 → 
  (2 * a * c - a * b * c : ℤ) ≤ 28 ∧ 
  ∃ a' b' c' : ℕ+, a' ≤ 7 ∧ b' ≤ 6 ∧ c' ≤ 4 ∧ 2 * a' * c' - a' * b' * c' = 28 :=
sorry

end NUMINAMATH_CALUDE_max_value_2ac_minus_abc_l1878_187837


namespace NUMINAMATH_CALUDE_panda_bamboo_consumption_l1878_187888

/-- The total weekly bamboo consumption for a group of pandas -/
def weekly_bamboo_consumption (small_pandas : ℕ) (big_pandas : ℕ) 
  (small_daily_consumption : ℕ) (big_daily_consumption : ℕ) : ℕ :=
  7 * ((small_pandas * small_daily_consumption) + (big_pandas * big_daily_consumption))

/-- Theorem stating the weekly bamboo consumption for 9 specific pandas -/
theorem panda_bamboo_consumption :
  weekly_bamboo_consumption 4 5 25 40 = 2100 := by
  sorry

#eval weekly_bamboo_consumption 4 5 25 40

end NUMINAMATH_CALUDE_panda_bamboo_consumption_l1878_187888


namespace NUMINAMATH_CALUDE_carpet_shaded_area_l1878_187851

/-- Represents the carpet configuration with shaded squares -/
structure CarpetConfig where
  carpet_side : ℝ
  large_square_side : ℝ
  small_square_side : ℝ
  large_square_count : ℕ
  small_square_count : ℕ

/-- Calculates the total shaded area of the carpet -/
def total_shaded_area (config : CarpetConfig) : ℝ :=
  config.large_square_count * config.large_square_side^2 +
  config.small_square_count * config.small_square_side^2

/-- Theorem stating the total shaded area of the carpet with given conditions -/
theorem carpet_shaded_area :
  ∀ (config : CarpetConfig),
    config.carpet_side = 12 →
    config.carpet_side / config.large_square_side = 4 →
    config.large_square_side / config.small_square_side = 3 →
    config.large_square_count = 1 →
    config.small_square_count = 8 →
    total_shaded_area config = 17 := by
  sorry


end NUMINAMATH_CALUDE_carpet_shaded_area_l1878_187851
