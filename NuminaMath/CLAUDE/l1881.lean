import Mathlib

namespace symmetric_difference_A_B_l1881_188138

/-- Set difference -/
def set_difference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}

/-- Symmetric difference -/
def symmetric_difference (M N : Set ℝ) : Set ℝ :=
  set_difference M N ∪ set_difference N M

/-- Set A -/
def A : Set ℝ := {t | ∃ x, t = x^2 - 3*x}

/-- Set B -/
def B : Set ℝ := {x | ∃ y, y = Real.log (-x)}

theorem symmetric_difference_A_B :
  symmetric_difference A B = {x | x < -9/4 ∨ x ≥ 0} := by sorry

end symmetric_difference_A_B_l1881_188138


namespace complex_equation_solution_l1881_188166

theorem complex_equation_solution (z : ℂ) :
  z / (1 - Complex.I) = Complex.I ^ 2016 + Complex.I ^ 2017 → z = 2 := by
  sorry

end complex_equation_solution_l1881_188166


namespace grape_rate_calculation_l1881_188121

/-- The rate per kg of grapes -/
def grape_rate : ℝ := 70

/-- The amount of grapes purchased in kg -/
def grape_amount : ℝ := 10

/-- The rate per kg of mangoes -/
def mango_rate : ℝ := 55

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℝ := 9

/-- The total amount paid to the shopkeeper -/
def total_paid : ℝ := 1195

theorem grape_rate_calculation :
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
by sorry

end grape_rate_calculation_l1881_188121


namespace polynomial_value_l1881_188168

theorem polynomial_value (x : ℝ) (h : x^2 - 2*x + 6 = 9) : 2*x^2 - 4*x + 6 = 12 := by
  sorry

end polynomial_value_l1881_188168


namespace triangle_inequality_l1881_188161

theorem triangle_inequality (A B C : ℝ) (h : A + B + C = π) :
  Real.tan (B / 2) * Real.tan (C / 2) ≤ ((1 - Real.sin (A / 2)) / Real.cos (A / 2))^2 := by
  sorry

end triangle_inequality_l1881_188161


namespace mobile_phone_cost_l1881_188112

def refrigerator_cost : ℝ := 15000
def refrigerator_loss_percent : ℝ := 4
def mobile_profit_percent : ℝ := 9
def overall_profit : ℝ := 120

theorem mobile_phone_cost (mobile_cost : ℝ) : 
  (refrigerator_cost * (1 - refrigerator_loss_percent / 100) + 
   mobile_cost * (1 + mobile_profit_percent / 100)) - 
  (refrigerator_cost + mobile_cost) = overall_profit →
  mobile_cost = 8000 := by
sorry

end mobile_phone_cost_l1881_188112


namespace square_plus_one_ge_twice_abs_l1881_188123

theorem square_plus_one_ge_twice_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end square_plus_one_ge_twice_abs_l1881_188123


namespace sin_150_degrees_l1881_188167

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end sin_150_degrees_l1881_188167


namespace power_of_81_l1881_188159

theorem power_of_81 : (81 : ℝ) ^ (5/4 : ℝ) = 243 := by sorry

end power_of_81_l1881_188159


namespace f_negative_iff_a_greater_than_ten_no_integer_a_for_g_local_minimum_l1881_188114

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - a * x^2 + 8

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x + 4 * a * x^2 - 12 * a^2 * x + 3 * a^3 - 8

-- Theorem 1: f(x) < 0 for all x ∈ [1, 2] iff a > 10
theorem f_negative_iff_a_greater_than_ten (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a x < 0) ↔ a > 10 := by sorry

-- Theorem 2: No integer a exists such that g(x) has a local minimum in (0, 1)
theorem no_integer_a_for_g_local_minimum :
  ¬ ∃ a : ℤ, ∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ IsLocalMin (g (a : ℝ)) x := by sorry

end f_negative_iff_a_greater_than_ten_no_integer_a_for_g_local_minimum_l1881_188114


namespace wraps_percentage_increase_l1881_188184

/-- Given John's raw squat weight, the additional weight from sleeves, and the difference between wraps and sleeves, 
    calculate the percentage increase wraps provide to his raw squat. -/
theorem wraps_percentage_increase 
  (raw_squat : ℝ) 
  (sleeves_addition : ℝ) 
  (wraps_vs_sleeves_difference : ℝ) 
  (h1 : raw_squat = 600) 
  (h2 : sleeves_addition = 30) 
  (h3 : wraps_vs_sleeves_difference = 120) : 
  (raw_squat + sleeves_addition + wraps_vs_sleeves_difference - raw_squat) / raw_squat * 100 = 25 := by
sorry

end wraps_percentage_increase_l1881_188184


namespace repeating_decimal_fraction_l1881_188147

def repeating_decimal : ℚ := 36 / 99

theorem repeating_decimal_fraction :
  repeating_decimal = 4 / 11 ∧
  4 + 11 = 15 := by sorry

end repeating_decimal_fraction_l1881_188147


namespace sum_cube_plus_twice_sum_squares_l1881_188146

theorem sum_cube_plus_twice_sum_squares : (3 + 7)^3 + 2*(3^2 + 7^2) = 1116 := by
  sorry

end sum_cube_plus_twice_sum_squares_l1881_188146


namespace largest_square_area_l1881_188145

theorem largest_square_area (A B C : ℝ × ℝ) (h_right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (h_sum_squares : (B.1 - A.1)^2 + (B.2 - A.2)^2 + (C.1 - B.1)^2 + (C.2 - B.2)^2 + 2 * ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 500) :
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 125 := by
  sorry

#check largest_square_area

end largest_square_area_l1881_188145


namespace binary_sum_equals_141_l1881_188113

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number 1010101₂ -/
def binary1 : List Bool := [true, false, true, false, true, false, true]

/-- The second binary number 111000₂ -/
def binary2 : List Bool := [false, false, false, true, true, true]

/-- The sum of the two binary numbers in decimal -/
def sum_decimal : ℕ := binary_to_decimal binary1 + binary_to_decimal binary2

theorem binary_sum_equals_141 : sum_decimal = 141 := by
  sorry

end binary_sum_equals_141_l1881_188113


namespace exists_four_mutual_l1881_188125

-- Define a type for people
def Person : Type := ℕ

-- Define a relation for familiarity
def familiar : Person → Person → Prop := sorry

-- Define a group of 18 people
def group : Finset Person := sorry

-- Axiom: The group has exactly 18 people
axiom group_size : Finset.card group = 18

-- Axiom: Any two people are either familiar or unfamiliar
axiom familiar_or_unfamiliar (p q : Person) : p ∈ group → q ∈ group → p ≠ q → 
  familiar p q ∨ ¬familiar p q

-- Theorem to prove
theorem exists_four_mutual (group : Finset Person) 
  (h₁ : Finset.card group = 18) 
  (h₂ : ∀ p q : Person, p ∈ group → q ∈ group → p ≠ q → familiar p q ∨ ¬familiar p q) :
  ∃ (s : Finset Person), Finset.card s = 4 ∧ s ⊆ group ∧
    (∀ p q : Person, p ∈ s → q ∈ s → p ≠ q → familiar p q) ∨
    (∀ p q : Person, p ∈ s → q ∈ s → p ≠ q → ¬familiar p q) :=
sorry

end exists_four_mutual_l1881_188125


namespace scientific_notation_of_2720000_l1881_188110

theorem scientific_notation_of_2720000 :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    2720000 = a * (10 : ℝ) ^ n ∧
    a = 2.72 ∧ n = 6 := by
  sorry

end scientific_notation_of_2720000_l1881_188110


namespace tangency_condition_single_intersection_condition_l1881_188173

-- Define the line l: y = kx - 3k + 2
def line (k x : ℝ) : ℝ := k * x - 3 * k + 2

-- Define the curve C: (x-1)^2 + (y+1)^2 = 4
def curve (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 4

-- Define the domain of x for the curve
def x_domain (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1

-- Theorem for tangency condition
theorem tangency_condition (k : ℝ) : 
  (∃ x, x_domain x ∧ curve x (line k x) ∧ 
   (∀ x', x' ≠ x → ¬curve x' (line k x'))) ↔ 
  k = 5/12 :=
sorry

-- Theorem for single intersection condition
theorem single_intersection_condition (k : ℝ) :
  (∃! x, x_domain x ∧ curve x (line k x)) ↔ 
  (1/2 < k ∧ k ≤ 5/2) ∨ k = 5/12 :=
sorry

end tangency_condition_single_intersection_condition_l1881_188173


namespace fractional_equation_solution_l1881_188130

theorem fractional_equation_solution :
  ∀ x : ℝ, x ≠ 0 → x ≠ 3 → (2 / (x - 3) = 3 / x) ↔ x = 9 := by
sorry

end fractional_equation_solution_l1881_188130


namespace playground_length_is_687_5_l1881_188174

/-- A rectangular playground with given perimeter, breadth, and diagonal -/
structure Playground where
  perimeter : ℝ
  breadth : ℝ
  diagonal : ℝ

/-- The length of a rectangular playground -/
def length (p : Playground) : ℝ :=
  ((p.diagonal ^ 2) - (p.breadth ^ 2)) ^ (1/2)

/-- Theorem stating the length of the specific playground -/
theorem playground_length_is_687_5 (p : Playground) 
  (h1 : p.perimeter = 1200)
  (h2 : p.breadth = 500)
  (h3 : p.diagonal = 850) : 
  length p = 687.5 := by
  sorry

end playground_length_is_687_5_l1881_188174


namespace arithmetic_mean_not_less_than_harmonic_mean_l1881_188151

theorem arithmetic_mean_not_less_than_harmonic_mean :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ (a + b) / 2 ≥ 2 / (1/a + 1/b) :=
sorry

end arithmetic_mean_not_less_than_harmonic_mean_l1881_188151


namespace cynthia_potato_harvest_l1881_188143

theorem cynthia_potato_harvest :
  ∀ (P : ℕ),
  (P ≥ 13) →
  (P - 13) % 2 = 0 →
  ((P - 13) / 2 - 13 = 436) →
  P = 911 :=
by
  sorry

end cynthia_potato_harvest_l1881_188143


namespace standing_arrangements_eq_48_l1881_188180

/-- The number of different standing arrangements for 5 students in a row,
    given the specified conditions. -/
def standing_arrangements : ℕ :=
  let total_students : ℕ := 5
  let positions_for_A : ℕ := total_students - 1
  let remaining_positions : ℕ := total_students - 1
  let arrangements_for_D_and_E : ℕ := remaining_positions * (remaining_positions - 1) / 2
  positions_for_A * arrangements_for_D_and_E

/-- Theorem stating that the number of standing arrangements is 48. -/
theorem standing_arrangements_eq_48 : standing_arrangements = 48 := by
  sorry

#eval standing_arrangements  -- This should output 48

end standing_arrangements_eq_48_l1881_188180


namespace centromeres_equal_chromosomes_centromeres_necessarily_equal_chromosomes_l1881_188107

-- Define basic biological concepts
def Chromosome : Type := Unit
def Centromere : Type := Unit
def Cell : Type := Unit
def Ribosome : Type := Unit
def DNAMolecule : Type := Unit
def Chromatid : Type := Unit
def HomologousChromosome : Type := Unit

-- Define the properties
def has_ribosome (c : Cell) : Prop := sorry
def is_eukaryotic (c : Cell) : Prop := sorry
def number_of_centromeres (c : Cell) : ℕ := sorry
def number_of_chromosomes (c : Cell) : ℕ := sorry
def number_of_dna_molecules (c : Cell) : ℕ := sorry
def number_of_chromatids (c : Cell) : ℕ := sorry
def size_and_shape (h : HomologousChromosome) : ℕ := sorry

-- State the theorem
theorem centromeres_equal_chromosomes :
  ∀ (c : Cell), number_of_centromeres c = number_of_chromosomes c :=
sorry

-- State the conditions
axiom cells_with_ribosomes :
  ∃ (c : Cell), has_ribosome c ∧ ¬is_eukaryotic c

axiom dna_chromatid_ratio :
  ∀ (c : Cell), 
    (number_of_dna_molecules c = number_of_chromatids c) ∨
    (number_of_dna_molecules c = 1 ∧ number_of_chromatids c = 0)

axiom homologous_chromosomes_different :
  ∃ (h1 h2 : HomologousChromosome), size_and_shape h1 ≠ size_and_shape h2

-- The main theorem stating that the statement is false
theorem centromeres_necessarily_equal_chromosomes :
  ¬(∃ (c : Cell), number_of_centromeres c ≠ number_of_chromosomes c) :=
sorry

end centromeres_equal_chromosomes_centromeres_necessarily_equal_chromosomes_l1881_188107


namespace heesu_has_greatest_sum_l1881_188124

def sora_numbers : List ℕ := [4, 6]
def heesu_numbers : List ℕ := [7, 5]
def jiyeon_numbers : List ℕ := [3, 8]

def sum_list (l : List ℕ) : ℕ := l.sum

theorem heesu_has_greatest_sum :
  sum_list heesu_numbers > sum_list sora_numbers ∧
  sum_list heesu_numbers > sum_list jiyeon_numbers :=
by sorry

end heesu_has_greatest_sum_l1881_188124


namespace inner_hexagon_area_l1881_188162

/-- Given a hexagon ABCDEF with specific area properties, prove the area of the inner hexagon A₁B₁C₁D₁E₁F₁ -/
theorem inner_hexagon_area 
  (area_ABCDEF : ℝ) 
  (area_triangle : ℝ) 
  (area_shaded : ℝ) 
  (h1 : area_ABCDEF = 2010) 
  (h2 : area_triangle = 335) 
  (h3 : area_shaded = 670) : 
  area_ABCDEF - (6 * area_triangle + area_shaded) / 2 = 670 := by
sorry

end inner_hexagon_area_l1881_188162


namespace additional_license_plates_l1881_188182

def initial_first_letter : Nat := 5
def initial_second_letter : Nat := 3
def initial_first_number : Nat := 5
def initial_second_number : Nat := 5

def new_first_letter : Nat := 5
def new_second_letter : Nat := 4
def new_first_number : Nat := 7
def new_second_number : Nat := 5

def initial_combinations : Nat := initial_first_letter * initial_second_letter * initial_first_number * initial_second_number

def new_combinations : Nat := new_first_letter * new_second_letter * new_first_number * new_second_number

theorem additional_license_plates :
  new_combinations - initial_combinations = 325 := by
  sorry

end additional_license_plates_l1881_188182


namespace parallelogram_df_and_area_l1881_188165

/-- Represents a parallelogram ABCD with altitudes DE and DF -/
structure Parallelogram where
  -- Length of side DC
  dc : ℝ
  -- Length of EB (part of base AB)
  eb : ℝ
  -- Length of altitude DE
  de : ℝ
  -- Assumption that ABCD is a parallelogram
  is_parallelogram : True

/-- Properties of the parallelogram -/
def parallelogram_properties (p : Parallelogram) : Prop :=
  p.dc = 15 ∧ p.eb = 3 ∧ p.de = 5

/-- Theorem about the length of DF and the area of the parallelogram -/
theorem parallelogram_df_and_area (p : Parallelogram) 
  (h : parallelogram_properties p) :
  ∃ (df area : ℝ), df = 5 ∧ area = 75 := by
  sorry


end parallelogram_df_and_area_l1881_188165


namespace set_difference_example_l1881_188102

def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem set_difference_example :
  let A : Set ℕ := {2, 3}
  let B : Set ℕ := {1, 3, 4}
  set_difference A B = {2} := by
sorry

end set_difference_example_l1881_188102


namespace min_sum_max_product_l1881_188189

theorem min_sum_max_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a * b = 1 → a + b ≥ 2) ∧ (a + b = 1 → a * b ≤ 1/4) := by sorry

end min_sum_max_product_l1881_188189


namespace parabolas_intersection_circle_l1881_188198

/-- The parabolas y = (x + 2)^2 and x + 8 = (y - 2)^2 intersect at four points that lie on a circle with radius squared equal to 4 -/
theorem parabolas_intersection_circle (x y : ℝ) : 
  (y = (x + 2)^2 ∧ x + 8 = (y - 2)^2) → 
  ∃ (center_x center_y : ℝ), (x - center_x)^2 + (y - center_y)^2 = 4 :=
by sorry

end parabolas_intersection_circle_l1881_188198


namespace cube_cutting_l1881_188197

theorem cube_cutting (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (a^3 : ℕ) = 98 + b^3 → b = 3 := by
  sorry

end cube_cutting_l1881_188197


namespace complement_of_A_in_U_l1881_188139

def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1, 2}

theorem complement_of_A_in_U : Aᶜ = {3} := by
  sorry

end complement_of_A_in_U_l1881_188139


namespace club_member_ratio_l1881_188163

/-- 
Given a club with current members and additional members,
prove that the ratio of new total members to current members is 5:2.
-/
theorem club_member_ratio (current_members additional_members : ℕ) 
  (h1 : current_members = 10)
  (h2 : additional_members = 15) : 
  (current_members + additional_members) / current_members = 5 / 2 := by
  sorry

end club_member_ratio_l1881_188163


namespace perpendicular_line_m_value_l1881_188100

/-- Given a line passing through points (1, 2) and (m, 3) that is perpendicular
    to the line 2x - 3y + 1 = 0, prove that m = 1/3 -/
theorem perpendicular_line_m_value (m : ℝ) :
  let line1 := {(x, y) : ℝ × ℝ | 2*x - 3*y + 1 = 0}
  let line2 := {(x, y) : ℝ × ℝ | ∃ t : ℝ, x = 1 + t*(m - 1) ∧ y = 2 + t*(3 - 2)}
  (∀ (p q : ℝ × ℝ), p ∈ line1 → q ∈ line1 → p ≠ q →
    ∀ (r s : ℝ × ℝ), r ∈ line2 → s ∈ line2 → r ≠ s →
      (p.2 - q.2) * (r.1 - s.1) = -(p.1 - q.1) * (r.2 - s.2)) →
  m = 1/3 := by
sorry

end perpendicular_line_m_value_l1881_188100


namespace right_triangle_inradius_l1881_188103

/-- A right triangle with side lengths 5, 12, and 13 has an inradius of 2 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 5 ∧ b = 12 ∧ c = 13 →
  a^2 + b^2 = c^2 →
  r = (a * b) / (2 * (a + b + c)) →
  r = 2 := by
sorry

end right_triangle_inradius_l1881_188103


namespace f_properties_l1881_188134

def f (x : ℝ) : ℝ := x^2 * (x - 3) * (x + 1)

theorem f_properties :
  (∀ x, f x = f (-x)) ∧ 
  f (-1) = 0 ∧ 
  f 3 = 0 := by
sorry

end f_properties_l1881_188134


namespace first_divisor_problem_l1881_188194

theorem first_divisor_problem (n d : ℕ) : 
  n > 1 →
  n % d = 1 →
  n % 7 = 1 →
  (∀ m : ℕ, m > 1 ∧ m % d = 1 ∧ m % 7 = 1 → m ≥ n) →
  n = 175 →
  d = 29 := by
sorry

end first_divisor_problem_l1881_188194


namespace quadratic_factorization_l1881_188135

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end quadratic_factorization_l1881_188135


namespace license_plate_count_l1881_188149

/-- The number of digits used in the license plate -/
def num_digits : ℕ := 4

/-- The number of letters used in the license plate -/
def num_letters : ℕ := 3

/-- The number of possible digits (0-9) -/
def digit_choices : ℕ := 10

/-- The number of letters in the alphabet -/
def letter_choices : ℕ := 32

/-- The maximum number of different car license plates -/
def max_license_plates : ℕ := digit_choices ^ num_digits * letter_choices ^ num_letters

theorem license_plate_count : max_license_plates = 327680000 := by
  sorry

end license_plate_count_l1881_188149


namespace original_line_length_l1881_188199

-- Define the units
def cm : ℝ := 1
def meter : ℝ := 100 * cm

-- Define the problem parameters
def erased_length : ℝ := 10 * cm
def remaining_length : ℝ := 90 * cm

-- State the theorem
theorem original_line_length :
  ∃ (original_length : ℝ),
    original_length = remaining_length + erased_length ∧
    original_length = 1 * meter :=
by sorry

end original_line_length_l1881_188199


namespace jeremy_age_l1881_188171

theorem jeremy_age (total_age : ℕ) (amy_age : ℚ) (chris_age : ℚ) (jeremy_age : ℚ) : 
  total_age = 132 →
  amy_age = (1 : ℚ) / 3 * jeremy_age →
  chris_age = 2 * amy_age →
  jeremy_age + amy_age + chris_age = total_age →
  jeremy_age = 66 :=
by sorry

end jeremy_age_l1881_188171


namespace equation_value_l1881_188105

theorem equation_value (a b c d : ℝ) 
  (eq1 : a + b - c - d = 5)
  (eq2 : (b - d)^2 = 16) :
  (a - b - c + d = b + 3*d - 4) ∨ (a - b - c + d = b + 3*d + 4) := by
  sorry

end equation_value_l1881_188105


namespace amount_left_after_purchase_l1881_188120

/-- Represents the price of a single lollipop in dollars -/
def lollipop_price : ℚ := 3/2

/-- Represents the price of a pack of gummies in dollars -/
def gummies_price : ℚ := 2

/-- Represents the number of lollipops bought -/
def num_lollipops : ℕ := 4

/-- Represents the number of packs of gummies bought -/
def num_gummies : ℕ := 2

/-- Represents the initial amount of money Chastity had in dollars -/
def initial_amount : ℚ := 15

/-- Theorem stating that the amount left after purchasing the candies is $5 -/
theorem amount_left_after_purchase : 
  initial_amount - (↑num_lollipops * lollipop_price + ↑num_gummies * gummies_price) = 5 := by
  sorry

end amount_left_after_purchase_l1881_188120


namespace sam_has_two_nickels_l1881_188104

/-- Represents the types of coins in Sam's wallet -/
inductive Coin
  | Penny
  | Nickel
  | Dime

/-- The value of a coin in cents -/
def coinValue : Coin → Nat
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10

/-- Represents Sam's wallet -/
structure Wallet :=
  (pennies : Nat)
  (nickels : Nat)
  (dimes : Nat)

/-- The total value of coins in the wallet in cents -/
def totalValue (w : Wallet) : Nat :=
  w.pennies * coinValue Coin.Penny +
  w.nickels * coinValue Coin.Nickel +
  w.dimes * coinValue Coin.Dime

/-- The total number of coins in the wallet -/
def totalCoins (w : Wallet) : Nat :=
  w.pennies + w.nickels + w.dimes

/-- The average value of coins in the wallet in cents -/
def averageValue (w : Wallet) : Rat :=
  (totalValue w : Rat) / (totalCoins w : Rat)

theorem sam_has_two_nickels (w : Wallet) 
  (h1 : averageValue w = 15)
  (h2 : averageValue { pennies := w.pennies, nickels := w.nickels, dimes := w.dimes + 1 } = 16) :
  w.nickels = 2 := by
  sorry

end sam_has_two_nickels_l1881_188104


namespace trailing_zeros_28_factorial_l1881_188106

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- Theorem: The number of trailing zeros in 28! is 6 -/
theorem trailing_zeros_28_factorial :
  trailingZeros 28 = 6 := by
  sorry

end trailing_zeros_28_factorial_l1881_188106


namespace diophantine_equation_solution_l1881_188154

theorem diophantine_equation_solution :
  ∀ x y z : ℕ, 3^x + 4^y = 5^z →
    ((x = 2 ∧ y = 2 ∧ z = 2) ∨ (x = 0 ∧ y = 1 ∧ z = 1)) :=
by sorry

end diophantine_equation_solution_l1881_188154


namespace min_reciprocal_sum_min_reciprocal_sum_equality_l1881_188136

theorem min_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (1 / a + 1 / b + 1 / c) ≥ 3 := by
  sorry

theorem min_reciprocal_sum_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (1 / a + 1 / b + 1 / c = 3) ↔ (a = 1 ∧ b = 1 ∧ c = 1) := by
  sorry

end min_reciprocal_sum_min_reciprocal_sum_equality_l1881_188136


namespace book_arrangement_count_l1881_188128

/-- Represents the number of math books -/
def num_math_books : Nat := 4

/-- Represents the number of history books -/
def num_history_books : Nat := 4

/-- Represents the condition that a math book must be at each end -/
def math_books_at_ends : Nat := 2

/-- Represents the remaining math books to be placed -/
def remaining_math_books : Nat := num_math_books - math_books_at_ends

/-- Represents the arrangement of books satisfying all conditions -/
def valid_arrangement (n m : Nat) : Nat :=
  (n * (n - 1)) * (m.factorial) * (remaining_math_books.factorial)

/-- Theorem stating the number of valid arrangements -/
theorem book_arrangement_count :
  valid_arrangement num_math_books num_history_books = 576 := by
  sorry

end book_arrangement_count_l1881_188128


namespace m_divided_by_8_l1881_188119

theorem m_divided_by_8 (m : ℕ) (h : m = 16^2018) : m / 8 = 2^8069 := by
  sorry

end m_divided_by_8_l1881_188119


namespace incircle_tangent_concurrency_l1881_188164

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric concepts
variable (is_convex_quadrilateral : Point → Point → Point → Point → Prop)
variable (is_incircle : Circle → Point → Point → Point → Prop)
variable (center_of : Circle → Point)
variable (second_common_external_tangent_touches : Circle → Circle → Point → Point → Prop)
variable (line_through : Point → Point → Set Point)
variable (concurrent : Set Point → Set Point → Set Point → Prop)

-- State the theorem
theorem incircle_tangent_concurrency 
  (A B C D : Point) 
  (ωA ωB : Circle) 
  (I J K L : Point) :
  is_convex_quadrilateral A B C D →
  is_incircle ωA A C D →
  is_incircle ωB B C D →
  I = center_of ωA →
  J = center_of ωB →
  second_common_external_tangent_touches ωA ωB K L →
  concurrent (line_through A K) (line_through B L) (line_through I J) :=
by sorry

end incircle_tangent_concurrency_l1881_188164


namespace max_value_K_l1881_188183

/-- The maximum value of K for x₁, x₂, x₃, x₄ ∈ [0,1] --/
theorem max_value_K : 
  ∃ (K_max : ℝ), K_max = Real.sqrt 5 / 125 ∧ 
  ∀ (x₁ x₂ x₃ x₄ : ℝ), 
    0 ≤ x₁ ∧ x₁ ≤ 1 ∧ 
    0 ≤ x₂ ∧ x₂ ≤ 1 ∧ 
    0 ≤ x₃ ∧ x₃ ≤ 1 ∧ 
    0 ≤ x₄ ∧ x₄ ≤ 1 → 
    let K := |x₁ - x₂| * |x₁ - x₃| * |x₁ - x₄| * |x₂ - x₃| * |x₂ - x₄| * |x₃ - x₄|
    K ≤ K_max :=
by sorry

end max_value_K_l1881_188183


namespace complex_number_existence_l1881_188195

theorem complex_number_existence : ∃ z : ℂ, 
  (∃ r : ℝ, z + 5 / z = r) ∧ 
  (Complex.re (z + 3) = 2 * Complex.im z) ∧ 
  (z = (1 : ℝ) + (2 : ℝ) * Complex.I ∨ z = (-11 : ℝ) / 5 - (2 : ℝ) / 5 * Complex.I) :=
by sorry

end complex_number_existence_l1881_188195


namespace age_ratio_problem_l1881_188193

/-- Given Tom's current age t and Sara's current age s, prove that the number of years
    until their age ratio is 3:2 is 7, given the conditions on their past ages. -/
theorem age_ratio_problem (t s : ℕ) (h1 : t - 3 = 2 * (s - 3)) (h2 : t - 8 = 3 * (s - 8)) :
  ∃ x : ℕ, x = 7 ∧ (t + x : ℚ) / (s + x) = 3 / 2 := by
  sorry

end age_ratio_problem_l1881_188193


namespace sector_max_area_l1881_188176

/-- Given a sector with constant perimeter a, prove that the maximum area is a²/16
    and this occurs when the central angle α is 2. -/
theorem sector_max_area (a : ℝ) (h : a > 0) :
  ∃ (S : ℝ) (α : ℝ),
    S = a^2 / 16 ∧
    α = 2 ∧
    ∀ (S' : ℝ) (α' : ℝ),
      (∃ (r : ℝ), 2 * r + r * α' = a ∧ S' = r^2 * α' / 2) →
      S' ≤ S :=
by sorry

end sector_max_area_l1881_188176


namespace tenth_diagram_shading_l1881_188144

/-- Represents a square grid with a specific shading pattern -/
structure ShadedGrid (n : ℕ) where
  size : ℕ
  shaded_squares : ℕ
  h_size : size = n * n
  h_shaded : shaded_squares = (n - 1) * (n / 2) + n

/-- The fraction of shaded squares in the grid -/
def shaded_fraction (grid : ShadedGrid n) : ℚ :=
  grid.shaded_squares / grid.size

theorem tenth_diagram_shading :
  ∃ (grid : ShadedGrid 10), shaded_fraction grid = 11 / 20 := by
  sorry

end tenth_diagram_shading_l1881_188144


namespace yan_distance_ratio_l1881_188141

/-- Represents the scenario of Yan's journey between home and stadium. -/
structure YanJourney where
  w : ℝ  -- Yan's walking speed
  x : ℝ  -- Distance from Yan to his home
  y : ℝ  -- Distance from Yan to the stadium
  h_positive : w > 0 -- Assumption that walking speed is positive
  h_between : x > 0 ∧ y > 0 -- Assumption that Yan is between home and stadium

/-- The theorem stating the ratio of Yan's distances. -/
theorem yan_distance_ratio (j : YanJourney) : 
  j.y / j.w = j.x / j.w + (j.x + j.y) / (7 * j.w) → j.x / j.y = 3 / 4 :=
by sorry

end yan_distance_ratio_l1881_188141


namespace problem_solution_l1881_188101

theorem problem_solution : 
  let M : ℚ := 2013 / 3
  let N : ℚ := M / 3
  let X : ℚ := M + N
  X = 895 := by sorry

end problem_solution_l1881_188101


namespace find_n_l1881_188175

/-- The average of the first 7 positive multiples of 5 -/
def a : ℚ := (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7

/-- The median of the first 3 positive multiples of n -/
def b (n : ℕ) : ℚ := 2 * n

/-- Theorem stating that given the conditions, n must equal 10 -/
theorem find_n : ∃ (n : ℕ), n > 0 ∧ a^2 - (b n)^2 = 0 ∧ n = 10 := by sorry

end find_n_l1881_188175


namespace vegetables_amount_l1881_188148

def beef_initial : ℕ := 4
def beef_unused : ℕ := 1

def beef_used (initial unused : ℕ) : ℕ := initial - unused

def vegetables_used (beef : ℕ) : ℕ := 2 * beef

theorem vegetables_amount : vegetables_used (beef_used beef_initial beef_unused) = 6 := by
  sorry

end vegetables_amount_l1881_188148


namespace algebraic_expression_equality_l1881_188158

theorem algebraic_expression_equality (a b : ℝ) (h : 5 * a + 3 * b = -4) :
  2 * (a + b) + 4 * (2 * a + b + 2) = 0 := by
  sorry

end algebraic_expression_equality_l1881_188158


namespace probability_one_authentic_one_defective_l1881_188152

def total_products : ℕ := 5
def authentic_products : ℕ := 4
def defective_products : ℕ := 1

theorem probability_one_authentic_one_defective :
  (authentic_products * defective_products : ℚ) / (total_products.choose 2) = 2 / 5 := by
sorry

end probability_one_authentic_one_defective_l1881_188152


namespace triangle_inequality_l1881_188191

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- State the theorem
theorem triangle_inequality (t : Triangle) : 
  Real.sin t.A * Real.cos t.C + t.A * Real.cos t.B > 0 := by
  sorry

end triangle_inequality_l1881_188191


namespace locus_of_M_l1881_188118

/-- Represents a point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- The setup of the problem -/
structure Configuration :=
  (A B C : Point)
  (D : Point)
  (l : Line)
  (P Q N M L : Point)

/-- Condition that A, B, and C are collinear -/
def collinear (A B C : Point) : Prop := sorry

/-- Condition that a point is not on a line -/
def not_on_line (P : Point) (l : Line) : Prop := sorry

/-- Condition that two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- Condition that a line passes through a point -/
def passes_through (l : Line) (P : Point) : Prop := sorry

/-- Condition that a point is the foot of the perpendicular from another point to a line -/
def is_foot_of_perpendicular (M C : Point) (l : Line) : Prop := sorry

/-- The main theorem -/
theorem locus_of_M (config : Configuration) :
  collinear config.A config.B config.C →
  not_on_line config.D config.l →
  parallel (Line.mk 0 0 0) (Line.mk 0 0 0) →  -- CP parallel to AD
  parallel (Line.mk 0 0 0) (Line.mk 0 0 0) →  -- CQ parallel to BD
  is_foot_of_perpendicular config.M config.C (Line.mk 0 0 0) →  -- PQ line
  (config.C.x - config.N.x) / (config.A.x - config.N.x) = (config.C.x - config.B.x) / (config.A.x - config.C.x) →
  ∃ (l_M : Line),
    passes_through l_M config.L ∧
    parallel l_M (Line.mk 0 0 0) ∧  -- MN line
    ∀ (M : Point), passes_through l_M M ↔ 
      ∃ (D : Point), is_foot_of_perpendicular M config.C (Line.mk 0 0 0) :=
sorry

end locus_of_M_l1881_188118


namespace place_mat_length_l1881_188142

-- Define the table and place mat properties
def table_radius : ℝ := 5
def num_mats : ℕ := 8
def mat_width : ℝ := 1.5

-- Define the length of the place mat
def mat_length (y : ℝ) : Prop :=
  y = table_radius * Real.sqrt (2 - Real.sqrt 2)

-- Define the arrangement of the place mats
def mats_arrangement (y : ℝ) : Prop :=
  ∃ (chord_length : ℝ),
    chord_length = 2 * table_radius * Real.sin (Real.pi / (2 * num_mats)) ∧
    y = chord_length

-- Theorem statement
theorem place_mat_length :
  ∃ y : ℝ, mat_length y ∧ mats_arrangement y :=
sorry

end place_mat_length_l1881_188142


namespace trees_needed_l1881_188109

/-- Represents a rectangular playground -/
structure Playground where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangular playground -/
def perimeter (p : Playground) : ℕ := 2 * (p.length + p.width)

/-- Represents the planting scheme for trees -/
structure PlantingScheme where
  treeSpacing : ℕ
  alternateTrees : Bool

/-- Calculates the total number of trees needed for a given playground and planting scheme -/
def totalTrees (p : Playground) (scheme : PlantingScheme) : ℕ :=
  (perimeter p) / scheme.treeSpacing

/-- Theorem stating the total number of trees required for the given playground and planting scheme -/
theorem trees_needed (p : Playground) (scheme : PlantingScheme) :
  p.length = 150 ∧ p.width = 60 ∧ scheme.treeSpacing = 10 ∧ scheme.alternateTrees = true →
  totalTrees p scheme = 42 := by
  sorry

end trees_needed_l1881_188109


namespace sams_remaining_dimes_l1881_188172

/-- Given that Sam initially had 9 dimes and gave 7 dimes away, prove that he now has 2 dimes. -/
theorem sams_remaining_dimes (initial_dimes : ℕ) (dimes_given_away : ℕ) 
  (h1 : initial_dimes = 9)
  (h2 : dimes_given_away = 7) :
  initial_dimes - dimes_given_away = 2 := by
  sorry

end sams_remaining_dimes_l1881_188172


namespace sum_18_47_in_base5_l1881_188155

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_18_47_in_base5 :
  toBase5 (18 + 47) = [2, 3, 0] :=
sorry

end sum_18_47_in_base5_l1881_188155


namespace cupcake_problem_l1881_188192

theorem cupcake_problem (cupcake_cost : ℚ) (individual_payment : ℚ) :
  cupcake_cost = 3/2 →
  individual_payment = 9 →
  (2 * individual_payment) / cupcake_cost = 12 :=
by sorry

end cupcake_problem_l1881_188192


namespace binary_101110_equals_octal_56_l1881_188178

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

/-- The binary representation of 101110₂ -/
def binary_101110 : List Bool := [false, true, true, true, true, false]

theorem binary_101110_equals_octal_56 :
  decimal_to_octal (binary_to_decimal binary_101110) = [6, 5] :=
by sorry

end binary_101110_equals_octal_56_l1881_188178


namespace m_range_l1881_188132

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (2*m) - y^2 / (m-1) = 1 ∧ m > 0 ∧ m < 1/3

def q (m : ℝ) : Prop := ∃ (e : ℝ), ∃ (x y : ℝ), y^2 / 5 - x^2 / m = 1 ∧ 1 < e ∧ e < 2 ∧ m > 0 ∧ m < 15

-- State the theorem
theorem m_range :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (1/3 ≤ m ∧ m < 15) :=
by sorry

end m_range_l1881_188132


namespace square_sum_given_sum_square_and_product_l1881_188179

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 := by
  sorry

end square_sum_given_sum_square_and_product_l1881_188179


namespace greatest_common_multiple_9_15_less_120_l1881_188170

theorem greatest_common_multiple_9_15_less_120 : ∃ n : ℕ,
  n = 90 ∧
  9 ∣ n ∧
  15 ∣ n ∧
  n < 120 ∧
  ∀ m : ℕ, (9 ∣ m ∧ 15 ∣ m ∧ m < 120) → m ≤ n :=
by sorry

end greatest_common_multiple_9_15_less_120_l1881_188170


namespace solution_set_quadratic_inequality_l1881_188122

theorem solution_set_quadratic_inequality :
  let f : ℝ → ℝ := λ x => x^2 - 3*x + 2
  {x : ℝ | f x ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end solution_set_quadratic_inequality_l1881_188122


namespace remaining_quadrilateral_perimeter_l1881_188117

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a quadrilateral with side lengths a, b, c, and d -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The perimeter of a quadrilateral -/
def Quadrilateral.perimeter (q : Quadrilateral) : ℝ :=
  q.a + q.b + q.c + q.d

/-- Given an equilateral triangle ABC with side length 4 and a right isosceles triangle DBE
    with DB = EB = 1 cut from it, the perimeter of the remaining quadrilateral ACED is 10 + √2 -/
theorem remaining_quadrilateral_perimeter :
  let abc : Triangle := { a := 4, b := 4, c := 4 }
  let dbe : Triangle := { a := 1, b := 1, c := Real.sqrt 2 }
  let aced : Quadrilateral := { a := 4, b := 3, c := Real.sqrt 2, d := 3 }
  aced.perimeter = 10 + Real.sqrt 2 := by
  sorry

end remaining_quadrilateral_perimeter_l1881_188117


namespace solve_for_n_l1881_188181

theorem solve_for_n (Q s r k : ℝ) (h : Q = (s * r) / (1 + k) ^ n) :
  n = Real.log ((s * r) / Q) / Real.log (1 + k) :=
by sorry

end solve_for_n_l1881_188181


namespace door_lock_problem_l1881_188177

def num_buttons : ℕ := 10
def buttons_to_press : ℕ := 3
def time_per_attempt : ℕ := 2

def total_combinations : ℕ := (num_buttons.choose buttons_to_press)

theorem door_lock_problem :
  (total_combinations * time_per_attempt = 240) ∧
  ((1 + total_combinations) / 2 * time_per_attempt = 121) ∧
  (((60 / time_per_attempt) - 1 : ℚ) / total_combinations = 29 / 120) := by
  sorry

end door_lock_problem_l1881_188177


namespace number_puzzle_l1881_188153

theorem number_puzzle : ∃ x : ℝ, 3 * (2 * x + 13) = 93 := by
  sorry

end number_puzzle_l1881_188153


namespace wrapping_cost_calculation_l1881_188160

/-- Represents the number of boxes a roll of wrapping paper can wrap -/
structure WrapCapacity where
  shirt : ℕ
  xl : ℕ

/-- Represents the number of boxes to be wrapped -/
structure BoxesToWrap where
  shirt : ℕ
  xl : ℕ

/-- Calculates the total cost of wrapping paper needed -/
def totalCost (capacity : WrapCapacity) (boxes : BoxesToWrap) (price_per_roll : ℚ) : ℚ :=
  let rolls_needed_shirt := (boxes.shirt + capacity.shirt - 1) / capacity.shirt
  let rolls_needed_xl := (boxes.xl + capacity.xl - 1) / capacity.xl
  (rolls_needed_shirt + rolls_needed_xl : ℚ) * price_per_roll

theorem wrapping_cost_calculation 
  (capacity : WrapCapacity) 
  (boxes : BoxesToWrap) 
  (price_per_roll : ℚ) :
  capacity.shirt = 5 →
  capacity.xl = 3 →
  boxes.shirt = 20 →
  boxes.xl = 12 →
  price_per_roll = 4 →
  totalCost capacity boxes price_per_roll = 32 :=
sorry

end wrapping_cost_calculation_l1881_188160


namespace f_increasing_max_b_value_ln_2_bounds_l1881_188185

noncomputable def f (x : ℝ) := Real.exp x - Real.exp (-x) - 2 * x

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by sorry

noncomputable def g (b : ℝ) (x : ℝ) := f (2 * x) - 4 * b * f x

theorem max_b_value : 
  (∀ x : ℝ, x > 0 → g 2 x > 0) ∧ 
  (∀ b : ℝ, b > 2 → ∃ x : ℝ, x > 0 ∧ g b x ≤ 0) := by sorry

theorem ln_2_bounds : 0.692 < Real.log 2 ∧ Real.log 2 < 0.694 := by sorry

end f_increasing_max_b_value_ln_2_bounds_l1881_188185


namespace initial_candies_l1881_188186

theorem initial_candies (package_size : ℕ) (added_candies : ℕ) (total_candies : ℕ) :
  package_size = 15 →
  added_candies = 4 →
  total_candies = 10 →
  total_candies - added_candies = 6 := by
  sorry

end initial_candies_l1881_188186


namespace emily_vacation_days_l1881_188116

/-- The number of days food lasts for dogs -/
def vacation_days (num_dogs : ℕ) (food_per_dog : ℕ) (total_food : ℕ) : ℕ :=
  total_food * 1000 / (num_dogs * food_per_dog)

/-- Theorem: Emily's vacation lasts 14 days -/
theorem emily_vacation_days :
  vacation_days 4 250 14 = 14 := by
  sorry

end emily_vacation_days_l1881_188116


namespace quadrilateral_area_l1881_188115

-- Define a structure for the partitioned triangle
structure PartitionedTriangle where
  -- Areas of the three smaller triangles
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  -- Area of the quadrilateral
  areaQuad : ℝ
  -- Total area of the original triangle
  totalArea : ℝ
  -- Condition: The sum of all areas equals the total area
  sum_areas : area1 + area2 + area3 + areaQuad = totalArea

-- Theorem statement
theorem quadrilateral_area (t : PartitionedTriangle) 
  (h1 : t.area1 = 4) 
  (h2 : t.area2 = 8) 
  (h3 : t.area3 = 12) : 
  t.areaQuad = 30 := by
  sorry

end quadrilateral_area_l1881_188115


namespace arithmetic_sequence_properties_l1881_188108

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) ∧
  (∀ n : ℕ, a n < a (n + 1)) ∧
  (a 3)^2 - 10 * (a 3) + 16 = 0 ∧
  (a 6)^2 - 10 * (a 6) + 16 = 0

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  (∀ n : ℕ, a n = 2 * n - 4) ∧ (a 136 = 268) := by
  sorry

end arithmetic_sequence_properties_l1881_188108


namespace power_of_five_times_112_l1881_188188

theorem power_of_five_times_112 : (112 * 5^4) = 70000 := by
  sorry

end power_of_five_times_112_l1881_188188


namespace cube_sum_squares_l1881_188133

theorem cube_sum_squares (a b t : ℝ) (h : a + b = t^2) :
  ∃ x y z : ℝ, 2 * (a^3 + b^3) = x^2 + y^2 + z^2 := by
  sorry

end cube_sum_squares_l1881_188133


namespace december_23_is_saturday_l1881_188150

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day of the week
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ k => nextDay (advanceDay d k)

-- Theorem statement
theorem december_23_is_saturday (thanksgiving : DayOfWeek) 
  (h : thanksgiving = DayOfWeek.Thursday) : 
  advanceDay thanksgiving 30 = DayOfWeek.Saturday := by
  sorry


end december_23_is_saturday_l1881_188150


namespace dividend_calculation_l1881_188157

theorem dividend_calculation (quotient divisor remainder : ℕ) 
  (h1 : quotient = 36)
  (h2 : divisor = 85)
  (h3 : remainder = 26) :
  divisor * quotient + remainder = 3086 := by
sorry

end dividend_calculation_l1881_188157


namespace constant_term_product_l1881_188137

-- Define polynomials p, q, and r
variable (p q r : ℝ[X])

-- Define the conditions
axiom h1 : r = p * q
axiom h2 : p.coeff 0 = 5
axiom h3 : p.leadingCoeff = 2
axiom h4 : p.degree = 2
axiom h5 : r.coeff 0 = -15

-- Theorem statement
theorem constant_term_product :
  q.eval 0 = -3 :=
sorry

end constant_term_product_l1881_188137


namespace tan_a_plus_pi_third_l1881_188140

theorem tan_a_plus_pi_third (a : Real) (h : Real.tan a = Real.sqrt 3) : 
  Real.tan (a + π/3) = -Real.sqrt 3 := by
  sorry

end tan_a_plus_pi_third_l1881_188140


namespace complement_of_A_l1881_188156

def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

theorem complement_of_A : Set.compl A = Set.Icc (-1) 3 := by sorry

end complement_of_A_l1881_188156


namespace value_of_a_l1881_188111

theorem value_of_a (a b d : ℤ) 
  (h1 : a + b = d) 
  (h2 : b + d = 7) 
  (h3 : d = 4) : 
  a = 1 := by
  sorry

end value_of_a_l1881_188111


namespace ricardo_coin_difference_l1881_188131

theorem ricardo_coin_difference :
  ∀ (one_cent five_cent : ℕ),
    one_cent + five_cent = 2020 →
    one_cent ≥ 1 →
    five_cent ≥ 1 →
    (5 * 2019 + 1) - (2019 + 5) = 8072 :=
by
  sorry

end ricardo_coin_difference_l1881_188131


namespace decreasing_condition_passes_through_origin_l1881_188190

/-- Given linear function y = (2-k)x - k^2 + 4 -/
def y (k x : ℝ) : ℝ := (2 - k) * x - k^2 + 4

/-- y decreases as x increases iff k > 2 -/
theorem decreasing_condition (k : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → y k x₁ > y k x₂) ↔ k > 2 :=
sorry

/-- The graph passes through the origin iff k = -2 -/
theorem passes_through_origin (k : ℝ) :
  y k 0 = 0 ↔ k = -2 :=
sorry

end decreasing_condition_passes_through_origin_l1881_188190


namespace simplify_expression_l1881_188196

theorem simplify_expression (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (9 * x^2 * y^3) / (12 * x * y^2) = 9 := by
  sorry

end simplify_expression_l1881_188196


namespace prob_third_white_specific_urn_l1881_188169

/-- An urn with white and black balls -/
structure Urn where
  white : ℕ
  black : ℕ

/-- The probability of drawing a white ball as the third ball -/
def prob_third_white (u : Urn) : ℚ :=
  u.white / (u.white + u.black)

/-- The theorem statement -/
theorem prob_third_white_specific_urn :
  let u : Urn := ⟨6, 5⟩
  prob_third_white u = 6 / 11 := by
  sorry

#eval prob_third_white ⟨6, 5⟩

end prob_third_white_specific_urn_l1881_188169


namespace laptop_cost_proof_l1881_188187

/-- The cost of the laptop satisfies the given conditions -/
theorem laptop_cost_proof (monthly_installment : ℝ) (down_payment_percent : ℝ) 
  (additional_down_payment : ℝ) (months_paid : ℕ) (remaining_balance : ℝ) :
  monthly_installment = 65 →
  down_payment_percent = 0.2 →
  additional_down_payment = 20 →
  months_paid = 4 →
  remaining_balance = 520 →
  ∃ (cost : ℝ), 
    cost - (down_payment_percent * cost + additional_down_payment + monthly_installment * months_paid) = remaining_balance ∧
    cost = 1000 := by
  sorry

end laptop_cost_proof_l1881_188187


namespace problem_1_l1881_188127

theorem problem_1 : (-2)^2 + Real.sqrt 12 - 2 * Real.sin (π / 3) = 4 + Real.sqrt 3 := by
  sorry

end problem_1_l1881_188127


namespace corner_cut_cube_edges_l1881_188129

/-- Represents a solid formed by removing smaller cubes from the corners of a larger cube -/
structure CornerCutCube where
  original_side_length : ℝ
  removed_side_length : ℝ

/-- Calculates the number of edges in the resulting solid -/
def edge_count (c : CornerCutCube) : ℕ :=
  sorry

/-- Theorem stating that a cube of side length 5 with corners of side length 2 removed has 48 edges -/
theorem corner_cut_cube_edges :
  let c : CornerCutCube := { original_side_length := 5, removed_side_length := 2 }
  edge_count c = 48 := by
  sorry

end corner_cut_cube_edges_l1881_188129


namespace stickers_needed_for_both_prizes_l1881_188126

def current_stickers : ℕ := 250
def small_prize_requirement : ℕ := 800
def big_prize_requirement : ℕ := 1500

theorem stickers_needed_for_both_prizes :
  (small_prize_requirement - current_stickers) + (big_prize_requirement - current_stickers) = 1800 :=
by sorry

end stickers_needed_for_both_prizes_l1881_188126
