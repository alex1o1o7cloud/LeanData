import Mathlib

namespace olivia_payment_l1928_192870

/-- Represents the number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- Represents the number of quarters Olivia pays for chips -/
def quarters_for_chips : ℕ := 4

/-- Represents the number of quarters Olivia pays for soda -/
def quarters_for_soda : ℕ := 12

/-- Calculates the total amount Olivia pays in dollars -/
def total_paid : ℚ :=
  (quarters_for_chips + quarters_for_soda) / quarters_per_dollar

theorem olivia_payment :
  total_paid = 4 := by sorry

end olivia_payment_l1928_192870


namespace quadratic_root_property_l1928_192853

theorem quadratic_root_property (m : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ + m = 0) → 
  (x₂^2 + 2*x₂ + m = 0) → 
  (x₁ + x₂ = x₁*x₂ - 1) → 
  m = -1 := by
sorry

end quadratic_root_property_l1928_192853


namespace square_area_l1928_192816

/-- Given a square with one vertex at (-6, -4) and diagonals intersecting at (3, 2),
    prove that its area is 234 square units. -/
theorem square_area (v : ℝ × ℝ) (c : ℝ × ℝ) (h1 : v = (-6, -4)) (h2 : c = (3, 2)) : 
  let d := 2 * Real.sqrt ((c.1 - v.1)^2 + (c.2 - v.2)^2)
  let s := d / Real.sqrt 2
  s^2 = 234 := by sorry

end square_area_l1928_192816


namespace sector_arc_length_l1928_192815

-- Define the sector
def Sector (area : ℝ) (angle : ℝ) : Type :=
  {r : ℝ // area = (1/2) * r^2 * angle}

-- Define the theorem
theorem sector_arc_length 
  (s : Sector 4 2) : 
  s.val * 2 = 4 := by sorry

end sector_arc_length_l1928_192815


namespace doctor_engineer_ratio_l1928_192872

theorem doctor_engineer_ratio (d l e : ℕ) (avg_age : ℚ) : 
  avg_age = 45 →
  (40 * d + 55 * l + 50 * e : ℚ) / (d + l + e : ℚ) = avg_age →
  d = 3 * e :=
by sorry

end doctor_engineer_ratio_l1928_192872


namespace inequality_proof_l1928_192808

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a - b)^2 / (2 * (a + b)) ≤ Real.sqrt ((a^2 + b^2) / 2) - Real.sqrt (a * b) ∧
  Real.sqrt ((a^2 + b^2) / 2) - Real.sqrt (a * b) ≤ (a - b)^2 / (Real.sqrt 2 * (a + b)) :=
by sorry

end inequality_proof_l1928_192808


namespace arun_lower_limit_l1928_192838

-- Define Arun's weight as a real number
variable (W : ℝ)

-- Define the conditions
def arun_upper_limit : Prop := W < 72
def brother_opinion : Prop := 60 < W ∧ W < 70
def mother_opinion : Prop := W ≤ 67
def average_weight : Prop := (W + 67) / 2 = 66

-- Define the theorem
theorem arun_lower_limit :
  arun_upper_limit W →
  brother_opinion W →
  mother_opinion W →
  average_weight W →
  W = 65 := by sorry

end arun_lower_limit_l1928_192838


namespace new_continental_math_institute_enrollment_l1928_192836

theorem new_continental_math_institute_enrollment :
  ∃! n : ℕ, n < 550 ∧ n % 19 = 15 ∧ n % 17 = 10 ∧ n = 509 := by
  sorry

end new_continental_math_institute_enrollment_l1928_192836


namespace system_solution_proof_l1928_192877

theorem system_solution_proof (x y : ℚ) : 
  (3 * x - y = 4 ∧ 6 * x - 3 * y = 10) ↔ (x = 2/3 ∧ y = -2) :=
by sorry

end system_solution_proof_l1928_192877


namespace batsman_average_after_12th_innings_l1928_192843

/-- Represents a batsman's performance over a series of innings -/
structure BatsmanPerformance where
  innings : ℕ
  lastInningsScore : ℕ
  averageIncrease : ℚ
  notOutCount : ℕ

/-- Calculates the average score after a given number of innings -/
def averageAfterInnings (performance : BatsmanPerformance) : ℚ :=
  sorry

theorem batsman_average_after_12th_innings 
  (performance : BatsmanPerformance)
  (h_innings : performance.innings = 12)
  (h_lastScore : performance.lastInningsScore = 60)
  (h_avgIncrease : performance.averageIncrease = 2)
  (h_notOut : performance.notOutCount = 0) :
  averageAfterInnings performance = 38 :=
sorry

end batsman_average_after_12th_innings_l1928_192843


namespace watch_sale_gain_percentage_l1928_192883

/-- Calculates the selling price given the cost price and loss percentage -/
def sellingPriceWithLoss (costPrice : ℚ) (lossPercentage : ℚ) : ℚ :=
  costPrice * (1 - lossPercentage / 100)

/-- Calculates the gain percentage given the cost price and selling price -/
def gainPercentage (costPrice : ℚ) (sellingPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

theorem watch_sale_gain_percentage 
  (costPrice : ℚ) 
  (lossPercentage : ℚ) 
  (additionalAmount : ℚ) : 
  costPrice = 3000 →
  lossPercentage = 10 →
  additionalAmount = 540 →
  gainPercentage costPrice (sellingPriceWithLoss costPrice lossPercentage + additionalAmount) = 8 := by
  sorry

end watch_sale_gain_percentage_l1928_192883


namespace smallest_a_for_divisibility_l1928_192869

theorem smallest_a_for_divisibility : 
  (∃ (a : ℕ), a > 0 ∧ 
    (∃ (n : ℕ), n > 0 ∧ Odd n ∧ 
      (2001 ∣ 55^n + a * 32^n))) ∧ 
  (∀ (a : ℕ), a > 0 → 
    (∃ (n : ℕ), n > 0 ∧ Odd n ∧ 
      (2001 ∣ 55^n + a * 32^n)) → 
    a ≥ 436) := by
  sorry

end smallest_a_for_divisibility_l1928_192869


namespace polynomial_division_remainder_l1928_192807

theorem polynomial_division_remainder (x : ℝ) : 
  x^1000 % ((x^2 + 1) * (x + 1)) = 1 := by sorry

end polynomial_division_remainder_l1928_192807


namespace closest_point_l1928_192829

def u (s : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 1 + 5*s
  | 1 => -2 + 3*s
  | 2 => -4 - 2*s

def b : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3
  | 1 => 3
  | 2 => 4

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 3
  | 2 => -2

theorem closest_point (s : ℝ) : 
  (∀ t : ℝ, ‖u s - b‖^2 ≤ ‖u t - b‖^2) ↔ s = 9/38 := by sorry

end closest_point_l1928_192829


namespace unique_solution_l1928_192841

theorem unique_solution (a b c : ℕ+) 
  (h1 : (Nat.gcd a.val b.val) = 1 ∧ (Nat.gcd b.val c.val) = 1 ∧ (Nat.gcd c.val a.val) = 1)
  (h2 : (a.val^2 + b.val) ∣ (b.val^2 + c.val))
  (h3 : (b.val^2 + c.val) ∣ (c.val^2 + a.val))
  (h4 : ∀ p : ℕ, Nat.Prime p → p ∣ (a.val^2 + b.val) → p % 7 ≠ 1) :
  a = 1 ∧ b = 1 ∧ c = 1 := by
sorry

end unique_solution_l1928_192841


namespace imaginary_part_of_z_l1928_192847

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (3 + i) / (1 + i)
  Complex.im z = -1 := by
sorry

end imaginary_part_of_z_l1928_192847


namespace quiz_true_false_count_l1928_192881

theorem quiz_true_false_count :
  ∀ n : ℕ,
  (2^n - 2) * 16 = 224 →
  n = 4 :=
by
  sorry

end quiz_true_false_count_l1928_192881


namespace plum_pies_count_l1928_192891

theorem plum_pies_count (total : ℕ) (ratio_r : ℕ) (ratio_p : ℕ) (ratio_m : ℕ) 
  (h_total : total = 30)
  (h_ratio : ratio_r = 2 ∧ ratio_p = 5 ∧ ratio_m = 3) :
  (total * ratio_m) / (ratio_r + ratio_p + ratio_m) = 9 := by
sorry

end plum_pies_count_l1928_192891


namespace line_equal_intercepts_l1928_192864

theorem line_equal_intercepts (a : ℝ) : 
  (∃ x y : ℝ, a * x + y - 2 - a = 0 ∧ 
   x = y ∧ 
   (x = 0 ∨ y = 0)) ↔ 
  (a = -2 ∨ a = 1) :=
sorry

end line_equal_intercepts_l1928_192864


namespace remaining_soup_feeds_twenty_adults_l1928_192862

/-- Represents the number of people a can of soup can feed -/
def people_per_can : ℕ := 4

/-- Represents the total number of cans of soup -/
def total_cans : ℕ := 10

/-- Represents the number of children fed -/
def children_fed : ℕ := 20

/-- Theorem: Given the conditions, prove that 20 adults can be fed with the remaining soup -/
theorem remaining_soup_feeds_twenty_adults :
  let cans_for_children := children_fed / people_per_can
  let remaining_cans := total_cans - cans_for_children
  remaining_cans * people_per_can = 20 := by
  sorry

#check remaining_soup_feeds_twenty_adults

end remaining_soup_feeds_twenty_adults_l1928_192862


namespace square_difference_theorem_l1928_192825

theorem square_difference_theorem (x y : ℚ) 
  (h1 : x + 2 * y = 5 / 9) 
  (h2 : x - 2 * y = 1 / 9) : 
  x^2 - 4 * y^2 = 5 / 81 := by
sorry

end square_difference_theorem_l1928_192825


namespace new_person_weight_l1928_192868

theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 5 →
  replaced_weight = 40 →
  avg_increase = 10 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 90 := by
  sorry

end new_person_weight_l1928_192868


namespace min_reciprocal_sum_l1928_192842

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 12) :
  (1 / a + 1 / b) ≥ (1 / 3) ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 12 ∧ 1 / a₀ + 1 / b₀ = 1 / 3 :=
by sorry

end min_reciprocal_sum_l1928_192842


namespace tan_seven_pi_fourths_l1928_192894

theorem tan_seven_pi_fourths : Real.tan (7 * π / 4) = -1 := by sorry

end tan_seven_pi_fourths_l1928_192894


namespace parallel_lines_distance_l1928_192844

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  a : ℝ
  l₁ : ℝ × ℝ → Prop := λ (x, y) ↦ 3 * x + 4 * y - 4 = 0
  l₂ : ℝ × ℝ → Prop := λ (x, y) ↦ a * x + 8 * y + 2 = 0
  parallel : ∃ (k : ℝ), a = 3 * k ∧ 8 = 4 * k

/-- The distance between two parallel lines -/
def distance (lines : ParallelLines) : ℝ :=
  1

/-- Theorem: The distance between the given parallel lines is 1 -/
theorem parallel_lines_distance (lines : ParallelLines) :
  distance lines = 1 := by sorry

end parallel_lines_distance_l1928_192844


namespace abscissa_range_theorem_l1928_192866

/-- The range of the abscissa of the center of circle M -/
def abscissa_range : Set ℝ := {a | a < 0 ∨ a > 12/5}

/-- The line on which the center of circle M lies -/
def center_line (x y : ℝ) : Prop := 2*x - y - 4 = 0

/-- The equation of circle M with center (a, 2a-4) and radius 1 -/
def circle_M (x y a : ℝ) : Prop := (x - a)^2 + (y - (2*a - 4))^2 = 1

/-- The condition that no point on circle M satisfies NO = 1/2 NA -/
def no_point_condition (x y : ℝ) : Prop := ¬(x^2 + y^2 = 1/4 * (x^2 + (y - 3)^2))

/-- The main theorem statement -/
theorem abscissa_range_theorem (a : ℝ) :
  (∀ x y : ℝ, center_line x y → circle_M x y a → no_point_condition x y) ↔ a ∈ abscissa_range :=
sorry

end abscissa_range_theorem_l1928_192866


namespace fraction_c_simplest_form_l1928_192855

def is_simplest_form (n d : ℤ) : Prop :=
  ∀ k : ℤ, k ≠ 0 → k ∣ n ∧ k ∣ d → k = 1 ∨ k = -1

theorem fraction_c_simplest_form (x y : ℤ) (hx : x ≠ 0) :
  is_simplest_form (x + y) (2 * x) :=
sorry

end fraction_c_simplest_form_l1928_192855


namespace boys_to_girls_ratio_l1928_192813

/-- Proves that in a college with 190 girls and 494 total students, the ratio of boys to girls is 152:95 -/
theorem boys_to_girls_ratio (total_students : ℕ) (girls : ℕ) 
  (h1 : total_students = 494) 
  (h2 : girls = 190) : 
  (total_students - girls) / girls = 152 / 95 := by
  sorry

end boys_to_girls_ratio_l1928_192813


namespace absolute_difference_inequality_l1928_192826

theorem absolute_difference_inequality (x : ℝ) :
  |2*x - 4| - |3*x + 9| < 1 ↔ x < -3 ∨ x > -6/5 := by
  sorry

end absolute_difference_inequality_l1928_192826


namespace quadratic_range_l1928_192857

theorem quadratic_range (a c : ℝ) (h1 : -4 ≤ a + c) (h2 : a + c ≤ -1)
  (h3 : -1 ≤ 4*a + c) (h4 : 4*a + c ≤ 5) : -1 ≤ 9*a + c ∧ 9*a + c ≤ 20 := by
  sorry

end quadratic_range_l1928_192857


namespace min_value_expression_l1928_192809

theorem min_value_expression (k n : ℝ) (h1 : k ≥ 0) (h2 : n ≥ 0) (h3 : 2 * k + n = 2) :
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 2 * x + y = 2 → 2 * k^2 - 4 * n ≤ 2 * x^2 - 4 * y ∧
  ∃ k₀ n₀ : ℝ, k₀ ≥ 0 ∧ n₀ ≥ 0 ∧ 2 * k₀ + n₀ = 2 ∧ 2 * k₀^2 - 4 * n₀ = -8 :=
sorry

end min_value_expression_l1928_192809


namespace divisibility_condition_pairs_l1928_192832

theorem divisibility_condition_pairs :
  ∀ m n : ℕ+,
  (∃ k : ℤ, (m : ℤ) + (n : ℤ)^2 = k * ((m : ℤ)^2 - (n : ℤ))) →
  (∃ l : ℤ, (n : ℤ) + (m : ℤ)^2 = l * ((n : ℤ)^2 - (m : ℤ))) →
  ((m = 2 ∧ n = 2) ∨ (m = 3 ∧ n = 3) ∨ (m = 1 ∧ n = 2) ∨
   (m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 3)) :=
by sorry

end divisibility_condition_pairs_l1928_192832


namespace three_equal_products_exist_l1928_192827

/-- Represents a 3x3 table filled with numbers from 1 to 9 --/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Checks if all numbers in the table are unique --/
def all_unique (t : Table) : Prop :=
  ∀ i j i' j', t i j = t i' j' → (i = i' ∧ j = j')

/-- Calculates the product of a row --/
def row_product (t : Table) (i : Fin 3) : ℕ :=
  ((t i 0).val + 1) * ((t i 1).val + 1) * ((t i 2).val + 1)

/-- Calculates the product of a column --/
def col_product (t : Table) (j : Fin 3) : ℕ :=
  ((t 0 j).val + 1) * ((t 1 j).val + 1) * ((t 2 j).val + 1)

/-- Checks if at least three products are equal --/
def three_equal_products (t : Table) : Prop :=
  ∃ p : ℕ, (
    (row_product t 0 = p ∧ row_product t 1 = p ∧ row_product t 2 = p) ∨
    (row_product t 0 = p ∧ row_product t 1 = p ∧ col_product t 0 = p) ∨
    (row_product t 0 = p ∧ row_product t 1 = p ∧ col_product t 1 = p) ∨
    (row_product t 0 = p ∧ row_product t 1 = p ∧ col_product t 2 = p) ∨
    (row_product t 0 = p ∧ row_product t 2 = p ∧ col_product t 0 = p) ∨
    (row_product t 0 = p ∧ row_product t 2 = p ∧ col_product t 1 = p) ∨
    (row_product t 0 = p ∧ row_product t 2 = p ∧ col_product t 2 = p) ∨
    (row_product t 0 = p ∧ col_product t 0 = p ∧ col_product t 1 = p) ∨
    (row_product t 0 = p ∧ col_product t 0 = p ∧ col_product t 2 = p) ∨
    (row_product t 0 = p ∧ col_product t 1 = p ∧ col_product t 2 = p) ∨
    (row_product t 1 = p ∧ row_product t 2 = p ∧ col_product t 0 = p) ∨
    (row_product t 1 = p ∧ row_product t 2 = p ∧ col_product t 1 = p) ∨
    (row_product t 1 = p ∧ row_product t 2 = p ∧ col_product t 2 = p) ∨
    (row_product t 1 = p ∧ col_product t 0 = p ∧ col_product t 1 = p) ∨
    (row_product t 1 = p ∧ col_product t 0 = p ∧ col_product t 2 = p) ∨
    (row_product t 1 = p ∧ col_product t 1 = p ∧ col_product t 2 = p) ∨
    (row_product t 2 = p ∧ col_product t 0 = p ∧ col_product t 1 = p) ∨
    (row_product t 2 = p ∧ col_product t 0 = p ∧ col_product t 2 = p) ∨
    (row_product t 2 = p ∧ col_product t 1 = p ∧ col_product t 2 = p) ∨
    (col_product t 0 = p ∧ col_product t 1 = p ∧ col_product t 2 = p)
  )

theorem three_equal_products_exist :
  ∃ t : Table, all_unique t ∧ three_equal_products t :=
by sorry

end three_equal_products_exist_l1928_192827


namespace unique_perfect_square_sum_l1928_192878

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def distinct_perfect_square_sum (a b c : ℕ) : Prop :=
  is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + c = 100

theorem unique_perfect_square_sum :
  ∃! (s : Finset (Finset ℕ)), s.card = 1 ∧
    ∀ t ∈ s, t.card = 3 ∧
      (∃ a b c, {a, b, c} = t ∧ distinct_perfect_square_sum a b c) :=
sorry

end unique_perfect_square_sum_l1928_192878


namespace water_transfer_height_l1928_192884

/-- Represents a rectangular tank with given dimensions -/
structure Tank where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of water in a tank given the water height -/
def waterVolume (t : Tank) (waterHeight : ℝ) : ℝ :=
  t.length * t.width * waterHeight

/-- Calculates the base area of a tank -/
def baseArea (t : Tank) : ℝ :=
  t.length * t.width

/-- Represents the problem setup -/
structure ProblemSetup where
  tankA : Tank
  tankB : Tank
  waterHeightB : ℝ

/-- The main theorem to prove -/
theorem water_transfer_height (setup : ProblemSetup) 
  (h1 : setup.tankA = { length := 4, width := 3, height := 5 })
  (h2 : setup.tankB = { length := 4, width := 2, height := 8 })
  (h3 : setup.waterHeightB = 1.5) :
  (waterVolume setup.tankB setup.waterHeightB) / (baseArea setup.tankA) = 1 := by
  sorry

end water_transfer_height_l1928_192884


namespace tan_600_l1928_192879

-- Define the tangent function (simplified for this example)
noncomputable def tan (x : ℝ) : ℝ := sorry

-- State the periodicity of tangent
axiom tan_periodic (x : ℝ) : tan (x + 180) = tan x

-- State the value of tan 60°
axiom tan_60 : tan 60 = Real.sqrt 3

-- Theorem to prove
theorem tan_600 : tan 600 = Real.sqrt 3 := by sorry

end tan_600_l1928_192879


namespace unique_solution_l1928_192887

/-- Represents a number of the form 13xy4.5z -/
def SpecialNumber (x y z : ℕ) : ℚ :=
  13000 + 100 * x + 10 * y + 4 + 0.5 + 0.01 * z

theorem unique_solution :
  ∃! (x y z : ℕ),
    x < 10 ∧ y < 10 ∧ z < 10 ∧
    (∃ (k : ℕ), SpecialNumber x y z = k * 792) ∧
    (45 + z) % 8 = 0 ∧
    (1 + 3 + x + y + 4 + 5 + z) % 9 = 0 ∧
    (1 - 3 + x - y + 4 - 5 + z) % 11 = 0 ∧
    SpecialNumber x y z = 13804.56 :=
by
  sorry

#eval SpecialNumber 8 0 6  -- Should output 13804.56

end unique_solution_l1928_192887


namespace compound_oxygen_count_l1928_192804

/-- Represents a chemical compound --/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ
  molecular_weight : ℕ

/-- Atomic weights of elements --/
def atomic_weight : ℕ → ℕ
  | 0 => 12  -- Carbon
  | 1 => 1   -- Hydrogen
  | 2 => 16  -- Oxygen
  | _ => 0   -- Other elements (not used in this problem)

/-- Calculate the molecular weight of a compound --/
def calculate_molecular_weight (c : Compound) : ℕ :=
  c.carbon * atomic_weight 0 + c.hydrogen * atomic_weight 1 + c.oxygen * atomic_weight 2

/-- Theorem: A compound with 4 Carbon, 8 Hydrogen, and molecular weight 88 has 2 Oxygen atoms --/
theorem compound_oxygen_count :
  ∀ c : Compound,
    c.carbon = 4 →
    c.hydrogen = 8 →
    c.molecular_weight = 88 →
    c.oxygen = 2 :=
by
  sorry

#check compound_oxygen_count

end compound_oxygen_count_l1928_192804


namespace olaf_boat_crew_size_l1928_192871

/-- Proves the number of men on Olaf's boat given the travel conditions -/
theorem olaf_boat_crew_size :
  ∀ (total_distance : ℝ) 
    (boat_speed : ℝ) 
    (water_per_man_per_day : ℝ) 
    (total_water : ℝ),
  total_distance = 4000 →
  boat_speed = 200 →
  water_per_man_per_day = 1/2 →
  total_water = 250 →
  (total_water / ((total_distance / boat_speed) * water_per_man_per_day) : ℝ) = 25 :=
by
  sorry


end olaf_boat_crew_size_l1928_192871


namespace minimum_point_of_translated_absolute_value_l1928_192800

-- Define the function
def f (x : ℝ) : ℝ := |x - 4| + 7

-- State the theorem
theorem minimum_point_of_translated_absolute_value :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x₀ ≤ f x) ∧ (f x₀ = 7) ∧ (x₀ = 4) := by
  sorry

end minimum_point_of_translated_absolute_value_l1928_192800


namespace max_quotient_value_l1928_192880

theorem max_quotient_value (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) :
  (∀ x y, 300 ≤ x ∧ x ≤ 500 → 800 ≤ y ∧ y ≤ 1600 → y / x ≤ 16 / 3) ∧
  (∃ x y, 300 ≤ x ∧ x ≤ 500 ∧ 800 ≤ y ∧ y ≤ 1600 ∧ y / x = 16 / 3) :=
by sorry

end max_quotient_value_l1928_192880


namespace two_digit_number_interchange_l1928_192806

theorem two_digit_number_interchange (x y : ℕ) : 
  x < 10 → y < 10 → x ≠ 0 → (10 * x + y) - (10 * y + x) = 54 → x - y = 6 := by
  sorry

end two_digit_number_interchange_l1928_192806


namespace girls_joined_l1928_192817

theorem girls_joined (initial_girls final_girls : ℕ) : 
  initial_girls = 732 → final_girls = 1414 → final_girls - initial_girls = 682 :=
by
  sorry

#check girls_joined

end girls_joined_l1928_192817


namespace quadratic_inequality_solution_set_l1928_192811

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-1/3 : ℝ) 2 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 + b * x + a < 0} = Set.Ioo (-3 : ℝ) (1/2) := by
  sorry

end quadratic_inequality_solution_set_l1928_192811


namespace potato_sale_revenue_l1928_192874

-- Define the given constants
def total_weight : ℕ := 6500
def damaged_weight : ℕ := 150
def bag_weight : ℕ := 50
def price_per_bag : ℕ := 72

-- Define the theorem
theorem potato_sale_revenue : 
  (((total_weight - damaged_weight) / bag_weight) * price_per_bag = 9144) := by
  sorry


end potato_sale_revenue_l1928_192874


namespace power_equality_n_equals_one_l1928_192828

theorem power_equality_n_equals_one :
  ∀ n : ℝ, (256 : ℝ) ^ (1/4 : ℝ) = 4 ^ n → n = 1 := by
sorry

end power_equality_n_equals_one_l1928_192828


namespace sin_seventeen_pi_quarters_l1928_192892

theorem sin_seventeen_pi_quarters : Real.sin (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end sin_seventeen_pi_quarters_l1928_192892


namespace quadratic_root_implies_u_equals_three_l1928_192837

theorem quadratic_root_implies_u_equals_three (u : ℝ) : 
  (6 * ((-19 + Real.sqrt 289) / 12)^2 + 19 * ((-19 + Real.sqrt 289) / 12) + u = 0) → u = 3 := by
  sorry

end quadratic_root_implies_u_equals_three_l1928_192837


namespace bromine_mass_percentage_not_37_21_l1928_192860

/-- The mass percentage of bromine in HBrO3 is not 37.21% -/
theorem bromine_mass_percentage_not_37_21 (H_mass Br_mass O_mass : ℝ) 
  (h1 : H_mass = 1.01)
  (h2 : Br_mass = 79.90)
  (h3 : O_mass = 16.00) :
  let HBrO3_mass := H_mass + Br_mass + 3 * O_mass
  (Br_mass / HBrO3_mass) * 100 ≠ 37.21 := by sorry

end bromine_mass_percentage_not_37_21_l1928_192860


namespace two_tangent_lines_l1928_192846

-- Define the function f(x) = x³ - x²
def f (x : ℝ) : ℝ := x^3 - x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x

-- Define a point of tangency
structure TangentPoint where
  x : ℝ
  y : ℝ
  slope : ℝ

-- Define a tangent line that passes through (1,0)
def isTangentLineThroughPoint (tp : TangentPoint) : Prop :=
  tp.y = f tp.x ∧ 
  tp.slope = f' tp.x ∧
  0 = tp.y + tp.slope * (1 - tp.x)

-- Theorem: There are exactly 2 tangent lines to f(x) that pass through (1,0)
theorem two_tangent_lines : 
  ∃! (s : Finset TangentPoint), 
    (∀ tp ∈ s, isTangentLineThroughPoint tp) ∧ 
    s.card = 2 := by
  sorry

end two_tangent_lines_l1928_192846


namespace monotonic_sufficient_not_necessary_l1928_192823

/-- A cubic polynomial function -/
def f (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- Monotonicity of a function on ℝ -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y ∨ (∀ x y, x < y → f x > f y)

/-- Intersection with x-axis at exactly one point -/
def IntersectsOnce (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

/-- Theorem stating that monotonicity is sufficient but not necessary for intersecting x-axis once -/
theorem monotonic_sufficient_not_necessary (b c d : ℝ) :
  (Monotonic (f b c d) → IntersectsOnce (f b c d)) ∧
  ¬(IntersectsOnce (f b c d) → Monotonic (f b c d)) :=
sorry

end monotonic_sufficient_not_necessary_l1928_192823


namespace shaded_area_between_circles_l1928_192867

theorem shaded_area_between_circles (r₁ r₂ r₃ R : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 5) (h₃ : r₃ = 2)
  (h_external : R = (r₁ + r₂ + r₁ + r₂) / 2)
  (h_tangent : r₁ + r₂ = R - r₁ - r₂) :
  π * R^2 - π * r₁^2 - π * r₂^2 - π * r₃^2 = 36 * π :=
by sorry

end shaded_area_between_circles_l1928_192867


namespace quadratic_sum_of_coefficients_l1928_192830

/-- A quadratic function passing through (-1,0) and (3,0) with a minimum value of 28 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_minus_one : a * (-1)^2 + b * (-1) + c = 0
  passes_through_three : a * 3^2 + b * 3 + c = 0
  min_value : ∃ (x : ℝ), ∀ (y : ℝ), a * x^2 + b * x + c ≤ a * y^2 + b * y + c ∧ a * x^2 + b * x + c = 28

/-- The sum of coefficients of the quadratic function is 28 -/
theorem quadratic_sum_of_coefficients (f : QuadraticFunction) : f.a + f.b + f.c = 28 := by
  sorry

end quadratic_sum_of_coefficients_l1928_192830


namespace book_cost_problem_l1928_192882

theorem book_cost_problem (total_cost : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) :
  total_cost = 360 ∧ loss_percent = 0.15 ∧ gain_percent = 0.19 →
  ∃ (cost_loss cost_gain : ℝ),
    cost_loss + cost_gain = total_cost ∧
    cost_loss * (1 - loss_percent) = cost_gain * (1 + gain_percent) ∧
    cost_loss = 210 := by
  sorry

end book_cost_problem_l1928_192882


namespace a_2017_equals_2_l1928_192885

-- Define the sequence S_n
def S (n : ℕ) : ℕ := 2 * n - 1

-- Define the sequence a_n
def a (n : ℕ) : ℕ := S n - S (n - 1)

-- Theorem statement
theorem a_2017_equals_2 : a 2017 = 2 := by
  sorry

end a_2017_equals_2_l1928_192885


namespace building_height_is_270_l1928_192833

/-- Calculates the height of a building with specified story heights -/
def buildingHeight (totalStories : ℕ) (firstHalfHeight : ℕ) (heightIncrease : ℕ) : ℕ :=
  let firstHalfStories := totalStories / 2
  let secondHalfStories := totalStories - firstHalfStories
  let firstHalfTotalHeight := firstHalfStories * firstHalfHeight
  let secondHalfHeight := firstHalfHeight + heightIncrease
  let secondHalfTotalHeight := secondHalfStories * secondHalfHeight
  firstHalfTotalHeight + secondHalfTotalHeight

/-- Theorem: The height of a 20-story building with specified story heights is 270 feet -/
theorem building_height_is_270 :
  buildingHeight 20 12 3 = 270 :=
by
  sorry -- Proof goes here

end building_height_is_270_l1928_192833


namespace polar_midpoint_specific_case_l1928_192886

/-- The midpoint of a line segment in polar coordinates --/
def polar_midpoint (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: The midpoint of a line segment with endpoints (10, π/4) and (10, 3π/4) in polar coordinates is (5√2, π/2) --/
theorem polar_midpoint_specific_case :
  let (r, θ) := polar_midpoint 10 (π/4) 10 (3*π/4)
  r = 5 * Real.sqrt 2 ∧ θ = π/2 := by sorry

end polar_midpoint_specific_case_l1928_192886


namespace smallest_possible_total_l1928_192863

/-- Represents the number of students in each grade --/
structure GradeCount where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ

/-- The ratios given in the problem --/
def ninth_to_tenth_ratio : Rat := 7 / 4
def ninth_to_eleventh_ratio : Rat := 5 / 3

/-- The condition that the ratios are correct --/
def ratios_correct (gc : GradeCount) : Prop :=
  (gc.ninth : Rat) / gc.tenth = ninth_to_tenth_ratio ∧
  (gc.ninth : Rat) / gc.eleventh = ninth_to_eleventh_ratio

/-- The total number of students --/
def total_students (gc : GradeCount) : ℕ :=
  gc.ninth + gc.tenth + gc.eleventh

/-- The main theorem to prove --/
theorem smallest_possible_total : 
  ∃ (gc : GradeCount), ratios_correct gc ∧ 
    (∀ (gc' : GradeCount), ratios_correct gc' → total_students gc ≤ total_students gc') ∧
    total_students gc = 76 := by
  sorry

end smallest_possible_total_l1928_192863


namespace absolute_value_inequality_solution_set_l1928_192849

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| < 2} = Set.Ioo (-1) 3 := by sorry

end absolute_value_inequality_solution_set_l1928_192849


namespace grid_paths_eq_binom_binom_eq_factorial_div_grid_paths_eq_factorial_div_l1928_192848

/-- The number of paths from (0,0) to (n,m) on a grid where only north and east movements are allowed -/
def grid_paths (m n : ℕ) : ℕ := sorry

/-- The binomial coefficient -/
def binom (n k : ℕ) : ℕ := sorry

theorem grid_paths_eq_binom (m n : ℕ) :
  grid_paths m n = binom (m + n) m :=
sorry

theorem binom_eq_factorial_div (n k : ℕ) :
  binom n k = (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k))) :=
sorry

theorem grid_paths_eq_factorial_div (m n : ℕ) :
  grid_paths m n = (Nat.factorial (m + n)) / ((Nat.factorial m) * (Nat.factorial n)) :=
sorry

end grid_paths_eq_binom_binom_eq_factorial_div_grid_paths_eq_factorial_div_l1928_192848


namespace weekly_distance_increase_l1928_192852

/-- Calculates the weekly distance increase for marathon training --/
theorem weekly_distance_increase 
  (initial_distance : ℚ) 
  (target_distance : ℚ) 
  (training_weeks : ℕ) 
  (h1 : initial_distance = 2) 
  (h2 : target_distance = 20) 
  (h3 : training_weeks = 27) :
  (target_distance - initial_distance) / training_weeks = 2/3 := by
  sorry


end weekly_distance_increase_l1928_192852


namespace complex_power_difference_l1928_192801

theorem complex_power_difference (θ : ℝ) (x : ℂ) (n : ℕ) 
  (h1 : 0 < θ ∧ θ < π) 
  (h2 : x - 1/x = 2*Complex.I*Real.sin θ) 
  (h3 : n > 0) : 
  x^n - 1/x^n = 2*Complex.I*Real.sin (n*θ) := by
  sorry

end complex_power_difference_l1928_192801


namespace chick_count_product_l1928_192893

/-- Represents the state of chicks in a nest for a given week -/
structure ChickState :=
  (open_beak : ℕ)
  (growing_feathers : ℕ)

/-- The chick lifecycle in the nest -/
def chick_lifecycle : Prop :=
  ∃ (last_week this_week : ChickState),
    last_week.open_beak = 20 ∧
    last_week.growing_feathers = 14 ∧
    this_week.open_beak = 15 ∧
    this_week.growing_feathers = 11

/-- The theorem to be proved -/
theorem chick_count_product :
  chick_lifecycle →
  ∃ (two_weeks_ago next_week : ℕ),
    two_weeks_ago = 11 ∧
    next_week = 15 ∧
    two_weeks_ago * next_week = 165 :=
by
  sorry


end chick_count_product_l1928_192893


namespace arithmetic_geometric_ratio_l1928_192859

/-- Given an arithmetic sequence with a non-zero common difference,
    if the 2nd, 3rd, and 6th terms form a geometric sequence,
    then the common ratio of these three terms is 3. -/
theorem arithmetic_geometric_ratio (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  d ≠ 0 →  -- non-zero common difference
  (a 2) * (a 6) = (a 3)^2 →  -- 2nd, 3rd, and 6th terms form a geometric sequence
  (a 3) / (a 2) = 3 :=  -- common ratio is 3
by sorry

end arithmetic_geometric_ratio_l1928_192859


namespace trajectory_is_ellipse_l1928_192831

/-- The equation of the trajectory of point M -/
def trajectory_equation (x y : ℝ) : Prop :=
  10 * Real.sqrt (x^2 + y^2) = |3*x + 4*y - 12|

/-- The trajectory of point M is an ellipse -/
theorem trajectory_is_ellipse :
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), trajectory_equation x y ↔ 
    ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) :=
sorry

end trajectory_is_ellipse_l1928_192831


namespace quadratic_equation_roots_l1928_192824

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  2 * x₁^2 - 3 * x₁ + 1 = 0 ∧ 2 * x₂^2 - 3 * x₂ + 1 = 0 := by
  sorry

end quadratic_equation_roots_l1928_192824


namespace student_weight_l1928_192895

/-- Prove that the student's present weight is 71 kilograms -/
theorem student_weight (student_weight sister_weight : ℝ) 
  (h1 : student_weight - 5 = 2 * sister_weight)
  (h2 : student_weight + sister_weight = 104) :
  student_weight = 71 := by
  sorry

end student_weight_l1928_192895


namespace interest_equality_l1928_192802

theorem interest_equality (P : ℝ) : 
  let I₁ := P * 0.04 * 5
  let I₂ := P * 0.05 * 4
  I₁ = I₂ ∧ I₁ = 20 := by sorry

end interest_equality_l1928_192802


namespace divisor_count_power_of_two_l1928_192845

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ+) : ℕ := sorry

/-- Number of divisors function -/
def num_of_divisors (n : ℕ+) : ℕ := sorry

/-- A natural number is a power of two -/
def is_power_of_two (n : ℕ) : Prop := sorry

theorem divisor_count_power_of_two (n : ℕ+) :
  is_power_of_two (sum_of_divisors n) → is_power_of_two (num_of_divisors n) := by
  sorry

end divisor_count_power_of_two_l1928_192845


namespace sunset_time_calculation_l1928_192810

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  m_valid : minutes < 60

def add_time_and_duration (t : Time) (d : Duration) : Time :=
  sorry

theorem sunset_time_calculation (sunrise : Time) (daylight : Duration) :
  sunrise.hours = 6 ∧ sunrise.minutes = 45 ∧
  daylight.hours = 11 ∧ daylight.minutes = 36 →
  let sunset := add_time_and_duration sunrise daylight
  sunset.hours = 18 ∧ sunset.minutes = 21 :=
sorry

end sunset_time_calculation_l1928_192810


namespace odd_periodic_function_value_l1928_192822

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_function_value 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 4) 
  (h_value : f (-1) = 2) : 
  f 13 = -2 := by
sorry

end odd_periodic_function_value_l1928_192822


namespace ball_selection_theorem_l1928_192890

def number_of_ways (total_red : ℕ) (total_white : ℕ) (balls_taken : ℕ) (min_score : ℕ) : ℕ :=
  let red_score := 2
  let white_score := 1
  (Finset.range (min total_red balls_taken + 1)).sum (fun red_taken =>
    let white_taken := balls_taken - red_taken
    if white_taken ≤ total_white ∧ red_taken * red_score + white_taken * white_score ≥ min_score
    then Nat.choose total_red red_taken * Nat.choose total_white white_taken
    else 0)

theorem ball_selection_theorem :
  number_of_ways 4 6 5 7 = 186 := by
  sorry

end ball_selection_theorem_l1928_192890


namespace bee_legs_count_l1928_192888

/-- Given 8 bees with a total of 48 legs, prove that each bee has 6 legs. -/
theorem bee_legs_count :
  let total_bees : ℕ := 8
  let total_legs : ℕ := 48
  total_legs / total_bees = 6 := by sorry

end bee_legs_count_l1928_192888


namespace simplify_expression_l1928_192897

theorem simplify_expression : (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) = (1 / 2) * (3^16 - 1) := by
  sorry

end simplify_expression_l1928_192897


namespace abs_2x_minus_1_lt_15_x_squared_plus_6x_minus_16_lt_0_abs_2x_plus_1_gt_13_x_squared_minus_2x_gt_0_l1928_192896

-- Question 1
theorem abs_2x_minus_1_lt_15 (x : ℝ) : 
  |2*x - 1| < 15 ↔ -7 < x ∧ x < 8 := by sorry

-- Question 2
theorem x_squared_plus_6x_minus_16_lt_0 (x : ℝ) : 
  x^2 + 6*x - 16 < 0 ↔ -8 < x ∧ x < 2 := by sorry

-- Question 3
theorem abs_2x_plus_1_gt_13 (x : ℝ) : 
  |2*x + 1| > 13 ↔ x < -7 ∨ x > 6 := by sorry

-- Question 4
theorem x_squared_minus_2x_gt_0 (x : ℝ) : 
  x^2 - 2*x > 0 ↔ x < 0 ∨ x > 2 := by sorry

end abs_2x_minus_1_lt_15_x_squared_plus_6x_minus_16_lt_0_abs_2x_plus_1_gt_13_x_squared_minus_2x_gt_0_l1928_192896


namespace plane_count_l1928_192876

theorem plane_count (total_wings : ℕ) (wings_per_plane : ℕ) (h1 : total_wings = 108) (h2 : wings_per_plane = 2) :
  total_wings / wings_per_plane = 54 := by
  sorry

end plane_count_l1928_192876


namespace abs_value_sum_l1928_192834

theorem abs_value_sum (a b c : ℚ) : 
  (abs a = 2) → 
  (abs b = 2) → 
  (abs c = 3) → 
  (b < 0) → 
  (0 < a) → 
  ((a + b + c = 3) ∨ (a + b + c = -3)) := by
sorry

end abs_value_sum_l1928_192834


namespace range_of_m_for_fractional_equation_l1928_192821

/-- The range of m for which the equation m/(x-2) + 1 = x/(2-x) has a non-negative solution x, where x ≠ 2 -/
theorem range_of_m_for_fractional_equation (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 2 ∧ m / (x - 2) + 1 = x / (2 - x)) ↔ 
  (m ≤ 2 ∧ m ≠ -2) :=
sorry

end range_of_m_for_fractional_equation_l1928_192821


namespace sum_of_nth_row_sum_of_100th_row_l1928_192835

/-- Represents the sum of numbers in the nth row of the triangular array -/
def f (n : ℕ) : ℕ :=
  2^n - 2

/-- Theorem stating that f(n) correctly represents the sum of numbers in the nth row -/
theorem sum_of_nth_row (n : ℕ) : 
  f n = 2^n - 2 :=
sorry

/-- Corollary for the 100th row -/
theorem sum_of_100th_row : 
  f 100 = 2^100 - 2 :=
sorry

end sum_of_nth_row_sum_of_100th_row_l1928_192835


namespace number_of_shoppers_l1928_192889

theorem number_of_shoppers (isabella sam giselle : ℕ) (shoppers : ℕ) : 
  isabella = sam + 45 →
  isabella = giselle + 15 →
  giselle = 120 →
  (isabella + sam + giselle) / shoppers = 115 →
  shoppers = 3 := by
sorry

end number_of_shoppers_l1928_192889


namespace equation_solution_l1928_192805

theorem equation_solution : ∃! x : ℚ, (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ∧ x = -7/6 := by sorry

end equation_solution_l1928_192805


namespace quadratic_roots_imply_m_equals_three_l1928_192819

theorem quadratic_roots_imply_m_equals_three (a m : ℤ) :
  a ≠ 1 →
  (∃ x y : ℤ, x > 0 ∧ y > 0 ∧ 
    (a - 1) * x^2 - m * x + a = 0 ∧
    (a - 1) * y^2 - m * y + a = 0) →
  m = 3 := by
sorry

end quadratic_roots_imply_m_equals_three_l1928_192819


namespace chloe_carrots_total_l1928_192851

/-- The total number of carrots Chloe has after picking, throwing out, and picking more. -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (additional : ℕ) : ℕ :=
  initial - thrown_out + additional

/-- Theorem stating that given the specific numbers in the problem, 
    the total number of carrots is 101. -/
theorem chloe_carrots_total : 
  total_carrots 128 94 67 = 101 := by
  sorry

end chloe_carrots_total_l1928_192851


namespace student_marks_l1928_192850

theorem student_marks (M P C : ℕ) : 
  C = P + 20 → 
  (M + C) / 2 = 40 → 
  M + P = 60 := by
sorry

end student_marks_l1928_192850


namespace michael_earnings_l1928_192814

/-- Michael's earnings from selling paintings --/
theorem michael_earnings (large_price small_price : ℕ) (large_quantity small_quantity : ℕ) :
  large_price = 100 →
  small_price = 80 →
  large_quantity = 5 →
  small_quantity = 8 →
  large_price * large_quantity + small_price * small_quantity = 1140 :=
by sorry

end michael_earnings_l1928_192814


namespace max_digits_product_l1928_192873

theorem max_digits_product (a b : ℕ) : 
  1000 ≤ a ∧ a < 10000 → 10000 ≤ b ∧ b < 100000 → 
  a * b < 1000000000 :=
sorry

end max_digits_product_l1928_192873


namespace garden_scale_drawing_l1928_192875

/-- Represents the length in feet given a scale drawing measurement -/
def actualLength (scale : ℝ) (drawingLength : ℝ) : ℝ :=
  scale * drawingLength

theorem garden_scale_drawing :
  let scale : ℝ := 500  -- 1 inch represents 500 feet
  let drawingLength : ℝ := 6.5  -- length in the drawing is 6.5 inches
  actualLength scale drawingLength = 3250 := by
  sorry

end garden_scale_drawing_l1928_192875


namespace prime_triplets_l1928_192854

def is_prime_triplet (a b c : ℕ) : Prop :=
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧
  Nat.Prime (a - b - 8) ∧ Nat.Prime (b - c - 8)

theorem prime_triplets :
  ∀ a b c : ℕ, is_prime_triplet a b c ↔ (a = 23 ∧ b = 13 ∧ (c = 2 ∨ c = 3)) :=
sorry

end prime_triplets_l1928_192854


namespace odd_function_extension_l1928_192820

-- Define an odd function on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function for x < 0
def f_neg (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f x = x^2 + x

-- Theorem statement
theorem odd_function_extension
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_neg : f_neg f) :
  ∀ x, x > 0 → f x = -x^2 + x :=
sorry

end odd_function_extension_l1928_192820


namespace simple_interest_proof_l1928_192840

/-- Given a principal amount for which the compound interest at 5% per annum for 2 years is 56.375,
    prove that the simple interest at 5% per annum for 2 years is 55. -/
theorem simple_interest_proof (P : ℝ) : 
  P * ((1 + 0.05)^2 - 1) = 56.375 → P * 0.05 * 2 = 55 := by
  sorry

end simple_interest_proof_l1928_192840


namespace four_b_b_two_divisible_by_seven_l1928_192861

theorem four_b_b_two_divisible_by_seven (B : ℕ) : 
  B ≤ 9 → (4000 + 110 * B + 2) % 7 = 0 ↔ B = 4 := by
  sorry

end four_b_b_two_divisible_by_seven_l1928_192861


namespace lailas_test_scores_l1928_192898

theorem lailas_test_scores (first_four_score last_score : ℕ) : 
  (0 ≤ first_four_score ∧ first_four_score ≤ 100) →
  (0 ≤ last_score ∧ last_score ≤ 100) →
  (last_score > first_four_score) →
  ((4 * first_four_score + last_score) / 5 = 82) →
  (∃ possible_scores : Finset ℕ, 
    possible_scores.card = 4 ∧
    last_score ∈ possible_scores ∧
    ∀ s, s ∈ possible_scores → 
      (0 ≤ s ∧ s ≤ 100) ∧
      (∃ x : ℕ, (0 ≤ x ∧ x ≤ 100) ∧ 
                (s > x) ∧ 
                ((4 * x + s) / 5 = 82))) :=
by sorry

end lailas_test_scores_l1928_192898


namespace intersection_empty_iff_b_in_range_l1928_192803

def set_A : Set ℝ := {x | -2 < x ∧ x < 1/3}

def set_B (b : ℝ) : Set ℝ := {x | x^2 - 4*b*x + 3*b^2 < 0}

theorem intersection_empty_iff_b_in_range (b : ℝ) :
  set_A ∩ set_B b = ∅ ↔ b ≥ 1/3 ∨ b ≤ -2 ∨ b = 0 := by
  sorry

end intersection_empty_iff_b_in_range_l1928_192803


namespace i_minus_one_in_second_quadrant_l1928_192858

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem i_minus_one_in_second_quadrant :
  is_in_second_quadrant (Complex.I - 1) := by
  sorry

end i_minus_one_in_second_quadrant_l1928_192858


namespace triangle_max_area_l1928_192865

open Real

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  c = sqrt 2 →
  b = a * sin C + c * A →
  C = π / 4 →
  ∃ (S : ℝ), S ≤ (1 + sqrt 2) / 2 ∧
    ∀ (S' : ℝ), S' = 1 / 2 * a * b * sin C → S' ≤ S :=
by sorry

end triangle_max_area_l1928_192865


namespace hexagonal_quadratic_coefficient_l1928_192818

-- Define hexagonal numbers
def hexagonal (n : ℕ) : ℕ := 2 * n^2 - n

-- Define the general quadratic form for hexagonal numbers
def quadratic_form (a b c n : ℕ) : ℕ := a * n^2 + b * n + c

-- Theorem statement
theorem hexagonal_quadratic_coefficient :
  ∃ (b c : ℕ), ∀ (n : ℕ), n > 0 → hexagonal n = quadratic_form 3 b c n :=
sorry

end hexagonal_quadratic_coefficient_l1928_192818


namespace problem_1_problem_2_problem_2_eval_0_problem_2_eval_3_l1928_192856

theorem problem_1 : -(-2)^2 + |(-Real.sqrt 3)| - 2 * Real.sin (π / 3) + (1 / 2)⁻¹ = -2 := by sorry

theorem problem_2 (m : ℝ) (h : m ≠ 2 ∧ m ≠ -2) : 
  (m / (m - 2) - 2 * m / (m^2 - 4)) + m / (m + 2) = (2 * m^2 - 2 * m) / (m^2 - 4) := by sorry

theorem problem_2_eval_0 : 
  (0 : ℝ) / (0 - 2) - 2 * 0 / (0^2 - 4) + 0 / (0 + 2) = 0 := by sorry

theorem problem_2_eval_3 : 
  (3 : ℝ) / (3 - 2) - 2 * 3 / (3^2 - 4) + 3 / (3 + 2) = 12 / 5 := by sorry

end problem_1_problem_2_problem_2_eval_0_problem_2_eval_3_l1928_192856


namespace smallest_integer_in_consecutive_set_l1928_192839

theorem smallest_integer_in_consecutive_set : 
  ∀ (n : ℤ), 
  (n + 6 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) / 7)) → 
  n ≥ -1 :=
by
  sorry

end smallest_integer_in_consecutive_set_l1928_192839


namespace total_votes_calculation_l1928_192899

theorem total_votes_calculation (V : ℝ) 
  (h1 : 0.3 * V + (0.3 * V + 1760) = V) : V = 4400 := by
  sorry

end total_votes_calculation_l1928_192899


namespace systematic_sampling_prob_example_l1928_192812

/-- Represents the probability of selection in systematic sampling -/
def systematic_sampling_probability (sample_size : ℕ) (population_size : ℕ) : ℚ :=
  sample_size / population_size

/-- Theorem: In systematic sampling with a sample size of 15 and a population size of 152,
    the probability of each person being selected is 15/152 -/
theorem systematic_sampling_prob_example :
  systematic_sampling_probability 15 152 = 15 / 152 := by
  sorry

end systematic_sampling_prob_example_l1928_192812
