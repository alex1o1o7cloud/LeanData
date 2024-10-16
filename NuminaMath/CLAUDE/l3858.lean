import Mathlib

namespace NUMINAMATH_CALUDE_knight_seating_probability_correct_l3858_385846

/-- The probability of three knights seated at a round table with n chairs
    having empty chairs on both sides of each knight. -/
def knight_seating_probability (n : ℕ) : ℚ :=
  if n ≥ 6 then
    (n - 4 : ℚ) * (n - 5) / ((n - 1 : ℚ) * (n - 2))
  else
    0

/-- Theorem: The probability of three knights seated at a round table with n chairs (n ≥ 6)
    having empty chairs on both sides of each knight is (n-4)(n-5) / ((n-1)(n-2)). -/
theorem knight_seating_probability_correct (n : ℕ) (h : n ≥ 6) :
  knight_seating_probability n = (n - 4 : ℚ) * (n - 5) / ((n - 1 : ℚ) * (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_knight_seating_probability_correct_l3858_385846


namespace NUMINAMATH_CALUDE_initial_oil_fraction_l3858_385855

/-- Proves that the initial fraction of oil in the cylinder was 3/4 -/
theorem initial_oil_fraction (total_capacity : ℕ) (added_bottles : ℕ) (final_fraction : ℚ) :
  total_capacity = 80 →
  added_bottles = 4 →
  final_fraction = 4/5 →
  (total_capacity : ℚ) * final_fraction - added_bottles = (3/4 : ℚ) * total_capacity := by
  sorry

end NUMINAMATH_CALUDE_initial_oil_fraction_l3858_385855


namespace NUMINAMATH_CALUDE_problem_1_l3858_385889

theorem problem_1 : 99 * (118 + 4/5) + 99 * (-1/5) - 99 * (18 + 3/5) = 9900 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3858_385889


namespace NUMINAMATH_CALUDE_complex_fraction_equals_two_l3858_385881

theorem complex_fraction_equals_two (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = a*b) :
  (a^6 + b^6) / (a + b)^6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_two_l3858_385881


namespace NUMINAMATH_CALUDE_min_value_of_function_l3858_385807

theorem min_value_of_function (x : ℝ) (h : x ≥ 0) :
  (3 * x^2 + 9 * x + 20) / (7 * (2 + x)) ≥ 10 / 7 ∧
  ∃ y ≥ 0, (3 * y^2 + 9 * y + 20) / (7 * (2 + y)) = 10 / 7 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3858_385807


namespace NUMINAMATH_CALUDE_inequality_proof_l3858_385828

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  (a^2 + b^2 + c^2 ≥ 1/3) ∧ 
  (a^2/b + b^2/c + c^2/a ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3858_385828


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l3858_385832

theorem sqrt_sum_inequality (x y α : ℝ) :
  Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α) →
  x + y ≥ 2 * α := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l3858_385832


namespace NUMINAMATH_CALUDE_problem_solution_l3858_385882

def g (n : ℤ) : ℤ :=
  if n % 2 = 0 then n - 2 else 3 * n

theorem problem_solution (m : ℤ) (h1 : m % 2 = 0) (h2 : g (g (g m)) = 54) : m = 60 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3858_385882


namespace NUMINAMATH_CALUDE_fraction_equality_l3858_385899

theorem fraction_equality (m n : ℝ) (h : 1/m + 1/n = 7) : 14*m*n/(m+n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3858_385899


namespace NUMINAMATH_CALUDE_fraction_comparison_l3858_385817

theorem fraction_comparison 
  (a b c d : ℤ) 
  (hc : c ≠ 0) 
  (hd : d ≠ 0) : 
  (c = d ∧ a > b → (a : ℚ) / c > (b : ℚ) / d) ∧
  (a = b ∧ c < d → (a : ℚ) / c > (b : ℚ) / d) ∧
  (a > b ∧ c < d → (a : ℚ) / c > (b : ℚ) / d) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3858_385817


namespace NUMINAMATH_CALUDE_no_real_solutions_for_arithmetic_progression_l3858_385819

-- Define the property of being an arithmetic progression
def is_arithmetic_progression (x y z w : ℝ) : Prop :=
  y - x = z - y ∧ z - y = w - z

-- Theorem statement
theorem no_real_solutions_for_arithmetic_progression :
  ¬ ∃ (a b : ℝ), is_arithmetic_progression 15 a b (a * b) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_arithmetic_progression_l3858_385819


namespace NUMINAMATH_CALUDE_unique_m_solution_l3858_385864

theorem unique_m_solution : ∃! m : ℝ, (1 - m)^4 + 6*(1 - m)^3 + 8*(1 - m) = 16*m^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_solution_l3858_385864


namespace NUMINAMATH_CALUDE_water_consumption_proof_l3858_385804

/-- Represents the daily dosage of a medication -/
structure MedicationSchedule where
  name : String
  timesPerDay : ℕ

/-- Represents the adherence to medication schedule -/
structure Adherence where
  medication : MedicationSchedule
  missedDoses : ℕ

def waterPerDose : ℕ := 4

def daysInWeek : ℕ := 7

def numWeeks : ℕ := 2

def medicationA : MedicationSchedule := ⟨"A", 3⟩
def medicationB : MedicationSchedule := ⟨"B", 4⟩
def medicationC : MedicationSchedule := ⟨"C", 2⟩

def adherenceA : Adherence := ⟨medicationA, 1⟩
def adherenceB : Adherence := ⟨medicationB, 2⟩
def adherenceC : Adherence := ⟨medicationC, 2⟩

def totalWaterConsumed : ℕ := 484

/-- Theorem stating that the total water consumed with medications over two weeks is 484 ounces -/
theorem water_consumption_proof :
  (medicationA.timesPerDay + medicationB.timesPerDay + medicationC.timesPerDay) * daysInWeek * numWeeks * waterPerDose -
  (adherenceA.missedDoses + adherenceB.missedDoses + adherenceC.missedDoses) * waterPerDose = totalWaterConsumed := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_proof_l3858_385804


namespace NUMINAMATH_CALUDE_giyoons_chocolates_l3858_385811

theorem giyoons_chocolates (initial_friends : ℕ) (absent_friends : ℕ) (extra_per_person : ℕ) (leftover : ℕ) :
  initial_friends = 8 →
  absent_friends = 2 →
  extra_per_person = 1 →
  leftover = 4 →
  ∃ (total_chocolates : ℕ),
    total_chocolates = (initial_friends - absent_friends) * ((total_chocolates / initial_friends) + extra_per_person) + leftover ∧
    total_chocolates = 40 :=
by sorry

end NUMINAMATH_CALUDE_giyoons_chocolates_l3858_385811


namespace NUMINAMATH_CALUDE_sandwich_theorem_l3858_385890

def sandwich_problem (david_spent : ℝ) (ben_spent : ℝ) : Prop :=
  ben_spent = 1.5 * david_spent ∧
  david_spent = ben_spent - 15 ∧
  david_spent + ben_spent = 75

theorem sandwich_theorem :
  ∃ (david_spent : ℝ) (ben_spent : ℝ), sandwich_problem david_spent ben_spent :=
by
  sorry

end NUMINAMATH_CALUDE_sandwich_theorem_l3858_385890


namespace NUMINAMATH_CALUDE_unique_triple_sum_l3858_385803

theorem unique_triple_sum (x y z : ℕ) : 
  x ≤ y ∧ y ≤ z ∧ x^x + y^y + z^z = 3382 ↔ (x, y, z) = (1, 4, 5) :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_sum_l3858_385803


namespace NUMINAMATH_CALUDE_fraction_calculation_l3858_385837

theorem fraction_calculation : 
  (2 + 1/4 + 0.25) / (2 + 3/4 - 1/2) + (2 * 0.5) / (2 + 1/5 - 2/5) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3858_385837


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l3858_385824

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_of_expression : units_digit (7 * 17 * 1977 - 7^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l3858_385824


namespace NUMINAMATH_CALUDE_ages_product_l3858_385888

/-- Represents the ages of the individuals in the problem -/
structure Ages where
  thomas : ℕ
  roy : ℕ
  kelly : ℕ
  julia : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.thomas = ages.roy - 6 ∧
  ages.thomas = ages.kelly + 4 ∧
  ages.roy = ages.julia + 8 ∧
  ages.roy = ages.kelly + 4 ∧
  ages.roy + 2 = 3 * (ages.julia + 2) ∧
  ages.thomas + 2 = 2 * (ages.kelly + 2)

/-- The theorem to be proved -/
theorem ages_product (ages : Ages) :
  satisfies_conditions ages →
  (ages.roy + 2) * (ages.kelly + 2) * (ages.thomas + 2) = 576 := by
  sorry

end NUMINAMATH_CALUDE_ages_product_l3858_385888


namespace NUMINAMATH_CALUDE_domain_of_f_l3858_385892

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Iic 2 :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l3858_385892


namespace NUMINAMATH_CALUDE_left_of_kolya_l3858_385829

/-- The number of people in a class lineup -/
structure ClassLineup where
  total : ℕ
  leftOfSasha : ℕ
  rightOfSasha : ℕ
  rightOfKolya : ℕ
  leftOfKolya : ℕ

/-- Theorem stating the number of people to the left of Kolya -/
theorem left_of_kolya (c : ClassLineup)
  (h1 : c.leftOfSasha = 20)
  (h2 : c.rightOfSasha = 8)
  (h3 : c.rightOfKolya = 12)
  (h4 : c.total = c.leftOfSasha + c.rightOfSasha + 1)
  (h5 : c.total = c.leftOfKolya + c.rightOfKolya + 1) :
  c.leftOfKolya = 16 := by
  sorry

end NUMINAMATH_CALUDE_left_of_kolya_l3858_385829


namespace NUMINAMATH_CALUDE_sphere_volume_after_radius_increase_l3858_385865

theorem sphere_volume_after_radius_increase (initial_surface_area : ℝ) (radius_increase : ℝ) : 
  initial_surface_area = 256 * Real.pi → 
  radius_increase = 2 → 
  (4 / 3) * Real.pi * ((initial_surface_area / (4 * Real.pi))^(1/2) + radius_increase)^3 = (4000 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_after_radius_increase_l3858_385865


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3858_385845

theorem largest_prime_factor_of_expression : 
  ∃ p : ℕ, p.Prime ∧ p ∣ (20^3 + 15^4 - 10^5) ∧ 
  ∀ q : ℕ, q.Prime → q ∣ (20^3 + 15^4 - 10^5) → q ≤ p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3858_385845


namespace NUMINAMATH_CALUDE_expression_bounds_l3858_385806

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  2 + Real.sqrt 2 ≤ 
    Real.sqrt (a^2 + (1-b)^2 + 1) + Real.sqrt (b^2 + (1-c)^2 + 1) + 
    Real.sqrt (c^2 + (1-d)^2 + 1) + Real.sqrt (d^2 + (1-a)^2 + 1) ∧
  Real.sqrt (a^2 + (1-b)^2 + 1) + Real.sqrt (b^2 + (1-c)^2 + 1) + 
  Real.sqrt (c^2 + (1-d)^2 + 1) + Real.sqrt (d^2 + (1-a)^2 + 1) ≤ 4 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_expression_bounds_l3858_385806


namespace NUMINAMATH_CALUDE_orangeade_pricing_l3858_385884

/-- Orangeade pricing problem -/
theorem orangeade_pricing
  (orange_juice : ℝ)  -- Amount of orange juice (same for both days)
  (water_day1 : ℝ)    -- Amount of water on day 1
  (water_day2 : ℝ)    -- Amount of water on day 2
  (price_day1 : ℝ)    -- Price per glass on day 1
  (h1 : water_day1 = orange_juice)        -- Equal amounts of orange juice and water on day 1
  (h2 : water_day2 = 2 * orange_juice)    -- Twice the amount of water on day 2
  (h3 : price_day1 = 0.48)                -- Price per glass on day 1 is $0.48
  (h4 : (orange_juice + water_day1) * price_day1 = 
        (orange_juice + water_day2) * price_day2) -- Same revenue on both days
  : price_day2 = 0.32 :=
by sorry

end NUMINAMATH_CALUDE_orangeade_pricing_l3858_385884


namespace NUMINAMATH_CALUDE_abc_inequality_and_fraction_sum_l3858_385850

theorem abc_inequality_and_fraction_sum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^2 + b^2 + c^2 = 9) : 
  a * b * c ≤ 3 * Real.sqrt 3 ∧ 
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b)) > (a + b + c) / 3 :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_and_fraction_sum_l3858_385850


namespace NUMINAMATH_CALUDE_f_even_implies_increasing_l3858_385809

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 1

theorem f_even_implies_increasing (m : ℝ) :
  (∀ x, f m x = f m (-x)) →
  ∀ a b, 0 < a → a < b → f m a < f m b :=
by sorry

end NUMINAMATH_CALUDE_f_even_implies_increasing_l3858_385809


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3858_385825

/-- An arithmetic sequence with given first term and 17th term -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_term : a 1 = 2
  term_17 : a 17 = 66

/-- The general formula for the nth term of the sequence -/
def general_formula (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  4 * n - 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = general_formula seq n) ∧
  ¬ ∃ n, seq.a n = 88 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3858_385825


namespace NUMINAMATH_CALUDE_valid_permutation_exists_valid_permutation_32_valid_permutation_100_l3858_385836

/-- A permutation of numbers from 1 to n satisfying the required property -/
def ValidPermutation (n : ℕ) : Type :=
  { p : Fin n → Fin n // Function.Bijective p ∧
    ∀ i j k, i < j → j < k →
      (p i).val + (p k).val ≠ 2 * (p j).val }

/-- The theorem stating the existence of a valid permutation for any n -/
theorem valid_permutation_exists (n : ℕ) : Nonempty (ValidPermutation n) := by
  sorry

/-- The specific cases for n = 32 and n = 100 -/
theorem valid_permutation_32 : Nonempty (ValidPermutation 32) := by
  exact valid_permutation_exists 32

theorem valid_permutation_100 : Nonempty (ValidPermutation 100) := by
  exact valid_permutation_exists 100

end NUMINAMATH_CALUDE_valid_permutation_exists_valid_permutation_32_valid_permutation_100_l3858_385836


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l3858_385823

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents the possible cuts of the plywood -/
inductive PlywoodCut
  | Vertical
  | Horizontal
  | Mixed

theorem plywood_cut_perimeter_difference :
  let plywood : Rectangle := { width := 6, height := 9 }
  let possible_cuts : List PlywoodCut := [PlywoodCut.Vertical, PlywoodCut.Horizontal, PlywoodCut.Mixed]
  let cut_rectangles : PlywoodCut → Rectangle
    | PlywoodCut.Vertical => { width := 1, height := 9 }
    | PlywoodCut.Horizontal => { width := 1, height := 6 }
    | PlywoodCut.Mixed => { width := 2, height := 3 }
  let perimeters : List ℝ := possible_cuts.map (fun cut => perimeter (cut_rectangles cut))
  (∃ (max_perimeter min_perimeter : ℝ),
    max_perimeter ∈ perimeters ∧
    min_perimeter ∈ perimeters ∧
    max_perimeter = perimeters.maximum ∧
    min_perimeter = perimeters.minimum ∧
    max_perimeter - min_perimeter = 10) := by
  sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l3858_385823


namespace NUMINAMATH_CALUDE_min_value_with_product_constraint_l3858_385883

theorem min_value_with_product_constraint (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (product_constraint : x * y * z = 32) : 
  x^2 + 4*x*y + 4*y^2 + 2*z^2 ≥ 68 ∧ 
  (x^2 + 4*x*y + 4*y^2 + 2*z^2 = 68 ↔ x = 4 ∧ y = 2 ∧ z = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_with_product_constraint_l3858_385883


namespace NUMINAMATH_CALUDE_quadratic_solution_l3858_385897

-- Define the quadratic equation
def quadratic_equation (p q x : ℝ) : Prop := x^2 + p*x + q = 0

-- Define the conditions
theorem quadratic_solution (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) :
  (quadratic_equation p q (2*p) ∧ quadratic_equation p q (q/2)) →
  p = 1 ∧ q = -6 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3858_385897


namespace NUMINAMATH_CALUDE_stock_comparison_l3858_385812

/-- Represents the final value of a stock after two years of changes --/
def final_value (initial : ℚ) (change1 : ℚ) (change2 : ℚ) : ℚ :=
  initial * (1 + change1) * (1 + change2)

/-- The problem statement --/
theorem stock_comparison : 
  let A := final_value 150 0.1 (-0.05)
  let B := final_value 100 (-0.3) 0.1
  let C := final_value 50 0 0.08
  C < B ∧ B < A := by sorry

end NUMINAMATH_CALUDE_stock_comparison_l3858_385812


namespace NUMINAMATH_CALUDE_smallest_shadow_area_l3858_385879

/-- The smallest area of the shadow cast by a cube onto a plane -/
theorem smallest_shadow_area (a b : ℝ) (h : b > a) (h_pos : a > 0) :
  ∃ (shadow_area : ℝ), shadow_area = (a^2 * b^2) / (b - a)^2 ∧
  ∀ (other_area : ℝ), other_area ≥ shadow_area := by
  sorry

end NUMINAMATH_CALUDE_smallest_shadow_area_l3858_385879


namespace NUMINAMATH_CALUDE_number_divided_by_6_multiplied_by_12_equals_9_l3858_385853

theorem number_divided_by_6_multiplied_by_12_equals_9 (x : ℝ) : (x / 6) * 12 = 9 → x = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_6_multiplied_by_12_equals_9_l3858_385853


namespace NUMINAMATH_CALUDE_rectangle_area_l3858_385818

/-- Given a rectangle with length three times its width and diagonal y, prove its area is 3y²/10 -/
theorem rectangle_area (y : ℝ) (y_pos : y > 0) : 
  ∃ w : ℝ, w > 0 ∧ 
  w^2 + (3*w)^2 = y^2 ∧ 
  3 * w^2 = (3 * y^2) / 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3858_385818


namespace NUMINAMATH_CALUDE_gcf_of_180_240_45_l3858_385868

theorem gcf_of_180_240_45 : Nat.gcd 180 (Nat.gcd 240 45) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_240_45_l3858_385868


namespace NUMINAMATH_CALUDE_figurine_cost_calculation_l3858_385801

def brand_a_price : ℝ := 65
def brand_b_price : ℝ := 75
def num_brand_a : ℕ := 3
def num_brand_b : ℕ := 2
def num_figurines : ℕ := 8
def figurine_total_cost : ℝ := brand_b_price + 40

theorem figurine_cost_calculation :
  (figurine_total_cost / num_figurines : ℝ) = 14.375 := by sorry

end NUMINAMATH_CALUDE_figurine_cost_calculation_l3858_385801


namespace NUMINAMATH_CALUDE_cos_squared_half_angle_minus_pi_fourth_l3858_385856

theorem cos_squared_half_angle_minus_pi_fourth (α : Real) 
  (h : Real.sin α = 2/3) : 
  Real.cos (α/2 - π/4)^2 = 1/6 := by sorry

end NUMINAMATH_CALUDE_cos_squared_half_angle_minus_pi_fourth_l3858_385856


namespace NUMINAMATH_CALUDE_closest_to_sqrt_diff_l3858_385870

def options : List ℝ := [0.18, 0.19, 0.20, 0.21, 0.22]

theorem closest_to_sqrt_diff (x : ℝ) (hx : x ∈ options) :
  x = 0.21 ↔ ∀ y ∈ options, |Real.sqrt 68 - Real.sqrt 64 - x| ≤ |Real.sqrt 68 - Real.sqrt 64 - y| :=
sorry

end NUMINAMATH_CALUDE_closest_to_sqrt_diff_l3858_385870


namespace NUMINAMATH_CALUDE_sewage_treatment_equipment_costs_l3858_385805

theorem sewage_treatment_equipment_costs (a b : ℝ) : 
  (a - b = 3) → (3 * b - 2 * a = 3) → (a = 12 ∧ b = 9) :=
by sorry

end NUMINAMATH_CALUDE_sewage_treatment_equipment_costs_l3858_385805


namespace NUMINAMATH_CALUDE_product_simplification_l3858_385800

theorem product_simplification (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l3858_385800


namespace NUMINAMATH_CALUDE_remaining_money_l3858_385859

def initial_amount : ℕ := 91
def sweater_cost : ℕ := 24
def tshirt_cost : ℕ := 6
def shoes_cost : ℕ := 11

theorem remaining_money : 
  initial_amount - (sweater_cost + tshirt_cost + shoes_cost) = 50 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l3858_385859


namespace NUMINAMATH_CALUDE_equal_area_triangle_square_l3858_385877

/-- A square with vertices O, S, U, V -/
structure Square (O S U V : ℝ × ℝ) : Prop where
  is_square : true  -- We assume OSUV is a square without proving it

/-- The area of a triangle given three points -/
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

/-- The area of a square given its side length -/
def square_area (side : ℝ) : ℝ := side * side

theorem equal_area_triangle_square 
  (O S U V W : ℝ × ℝ) 
  (h_square : Square O S U V)
  (h_O : O = (0, 0))
  (h_U : U = (3, 3))
  (h_W : W = (3, 9)) : 
  triangle_area S V W = square_area 3 := by
  sorry

#check equal_area_triangle_square

end NUMINAMATH_CALUDE_equal_area_triangle_square_l3858_385877


namespace NUMINAMATH_CALUDE_samantha_sleep_hours_l3858_385873

/-- Represents a time of day in 24-hour format -/
structure TimeOfDay where
  hour : Nat
  minute : Nat
  is_valid : hour < 24 ∧ minute < 60

/-- Calculates the number of hours between two times -/
def hoursBetween (t1 t2 : TimeOfDay) : Nat :=
  if t2.hour ≥ t1.hour then
    t2.hour - t1.hour
  else
    24 + t2.hour - t1.hour

/-- Samantha's bedtime -/
def bedtime : TimeOfDay := {
  hour := 19,
  minute := 0,
  is_valid := by simp
}

/-- Samantha's wake-up time -/
def wakeupTime : TimeOfDay := {
  hour := 11,
  minute := 0,
  is_valid := by simp
}

theorem samantha_sleep_hours :
  hoursBetween bedtime wakeupTime = 16 := by sorry

end NUMINAMATH_CALUDE_samantha_sleep_hours_l3858_385873


namespace NUMINAMATH_CALUDE_y_satisfies_equation_l3858_385827

open Real

/-- The function y(x) -/
noncomputable def y (x : ℝ) : ℝ := 2 * (sin x / x) + cos x

/-- The differential equation -/
def differential_equation (y : ℝ → ℝ) (x : ℝ) : Prop :=
  x * sin x * deriv y x + (sin x - x * cos x) * y x = sin x * cos x - x

/-- Theorem stating that y satisfies the differential equation -/
theorem y_satisfies_equation : ∀ x : ℝ, x ≠ 0 → differential_equation y x := by
  sorry

end NUMINAMATH_CALUDE_y_satisfies_equation_l3858_385827


namespace NUMINAMATH_CALUDE_undefined_values_l3858_385838

theorem undefined_values (x : ℝ) :
  (x^2 - 21*x + 110 = 0) ↔ (x = 10 ∨ x = 11) := by sorry

end NUMINAMATH_CALUDE_undefined_values_l3858_385838


namespace NUMINAMATH_CALUDE_square_with_triangles_removed_l3858_385878

theorem square_with_triangles_removed (s x y : ℝ) 
  (h1 : s - 2*x = 15)
  (h2 : s - 2*y = 9)
  (h3 : x > 0)
  (h4 : y > 0) :
  4 * (1/2 * x * y) = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_square_with_triangles_removed_l3858_385878


namespace NUMINAMATH_CALUDE_apple_profit_percentage_l3858_385831

theorem apple_profit_percentage 
  (total_apples : ℝ)
  (first_portion : ℝ)
  (second_portion : ℝ)
  (second_profit : ℝ)
  (overall_profit : ℝ)
  (h1 : total_apples = 280)
  (h2 : first_portion = 0.4)
  (h3 : second_portion = 0.6)
  (h4 : first_portion + second_portion = 1)
  (h5 : second_profit = 0.3)
  (h6 : overall_profit = 0.26)
  : ∃ (first_profit : ℝ),
    first_profit * first_portion * total_apples + 
    second_profit * second_portion * total_apples = 
    overall_profit * total_apples ∧
    first_profit = 0.2 := by
sorry

end NUMINAMATH_CALUDE_apple_profit_percentage_l3858_385831


namespace NUMINAMATH_CALUDE_exactly_two_valid_sets_l3858_385887

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)  -- The first integer in the set
  (length : ℕ) -- The number of integers in the set

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- Predicate for a valid set according to our problem -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  s.length ≥ 3 ∧ sum_consecutive s = 21

theorem exactly_two_valid_sets :
  ∃! (s₁ s₂ : ConsecutiveSet), is_valid_set s₁ ∧ is_valid_set s₂ ∧ s₁ ≠ s₂ ∧
    ∀ (s : ConsecutiveSet), is_valid_set s → s = s₁ ∨ s = s₂ :=
sorry

end NUMINAMATH_CALUDE_exactly_two_valid_sets_l3858_385887


namespace NUMINAMATH_CALUDE_parallel_lines_point_on_circle_l3858_385808

def line1 (a b x y : ℝ) : Prop := (b + 2) * x + a * y + 4 = 0

def line2 (a b x y : ℝ) : Prop := a * x + (2 - b) * y - 3 = 0

def parallel (f g : ℝ → ℝ → Prop) : Prop := 
  ∀ x₁ y₁ x₂ y₂, f x₁ y₁ ∧ f x₂ y₂ → (x₁ ≠ x₂ → (y₁ - y₂) / (x₁ - x₂) = (y₂ - y₁) / (x₂ - x₁))

theorem parallel_lines_point_on_circle (a b : ℝ) :
  parallel (line1 a b) (line2 a b) → a^2 + b^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_point_on_circle_l3858_385808


namespace NUMINAMATH_CALUDE_equivalence_condition_l3858_385820

theorem equivalence_condition (a : ℝ) : 
  (∀ x : ℝ, (5 - x) / (x - 2) ≥ 0 ↔ -3 < x ∧ x < a) ↔ a > 5 :=
by sorry

end NUMINAMATH_CALUDE_equivalence_condition_l3858_385820


namespace NUMINAMATH_CALUDE_gcd_lcm_product_36_210_l3858_385869

theorem gcd_lcm_product_36_210 : Nat.gcd 36 210 * Nat.lcm 36 210 = 7560 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_36_210_l3858_385869


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3858_385872

theorem quadratic_inequality_solution_sets (a : ℝ) :
  let f := fun x => x^2 - (1 + a) * x + a
  (a = 2 → {x | f x > 0} = {x | x > 2 ∨ x < 1}) ∧
  (a > 1 → {x | f x > 0} = {x | x > a ∨ x < 1}) ∧
  (a = 1 → {x | f x > 0} = {x | x ≠ 1}) ∧
  (a < 1 → {x | f x > 0} = {x | x > 1 ∨ x < a}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3858_385872


namespace NUMINAMATH_CALUDE_angle_edc_measure_l3858_385862

theorem angle_edc_measure (y : ℝ) :
  let angle_bde : ℝ := 4 * y
  let angle_edc : ℝ := 3 * y
  angle_bde + angle_edc = 180 →
  angle_edc = 540 / 7 := by
sorry

end NUMINAMATH_CALUDE_angle_edc_measure_l3858_385862


namespace NUMINAMATH_CALUDE_incircle_radius_inscribed_triangle_l3858_385849

theorem incircle_radius_inscribed_triangle (r : ℝ) (α β γ : ℝ) (h1 : 0 < r) (h2 : 0 < α) (h3 : 0 < β) (h4 : 0 < γ) 
  (h5 : α + β + γ = π) (h6 : Real.tan α = 1/3) (h7 : Real.sin β * Real.sin γ = 1/Real.sqrt 10) : 
  ∃ ρ : ℝ, ρ = (r * Real.sqrt 10) / (1 + Real.sqrt 2 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_incircle_radius_inscribed_triangle_l3858_385849


namespace NUMINAMATH_CALUDE_product_no_x_squared_term_l3858_385875

theorem product_no_x_squared_term (a : ℝ) : 
  (∀ x : ℝ, (x + 1) * (x^2 - 2*a*x + a^2) = x^3 + (a^2 - 2*a)*x + a^2) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_product_no_x_squared_term_l3858_385875


namespace NUMINAMATH_CALUDE_article_cost_l3858_385863

/-- The cost of an article given specific profit conditions -/
theorem article_cost (C : ℝ) (S : ℝ) : 
  S = 1.25 * C → -- Original selling price (25% profit)
  (0.8 * C + 0.3 * (0.8 * C) = S - 6.3) → -- New cost and selling price with 30% profit
  C = 30 := by
sorry

end NUMINAMATH_CALUDE_article_cost_l3858_385863


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l3858_385813

/-- A quadratic function with axis of symmetry at x = 9 and p(6) = 2 -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x, p a b c (9 - x) = p a b c (9 + x)) →
  p a b c 6 = 2 →
  p a b c 12 = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l3858_385813


namespace NUMINAMATH_CALUDE_square_minus_self_sum_l3858_385858

theorem square_minus_self_sum : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_self_sum_l3858_385858


namespace NUMINAMATH_CALUDE_households_using_only_brand_A_l3858_385834

/-- The number of households that use only brand A soap -/
def only_brand_A : ℕ := 60

/-- The number of households that use only brand B soap -/
def only_brand_B : ℕ := 75

/-- The number of households that use both brand A and brand B soap -/
def both_brands : ℕ := 25

/-- The number of households that use neither brand A nor brand B soap -/
def neither_brand : ℕ := 80

/-- The total number of households surveyed -/
def total_households : ℕ := 240

/-- Theorem stating that the number of households using only brand A soap is 60 -/
theorem households_using_only_brand_A :
  only_brand_A = total_households - only_brand_B - both_brands - neither_brand :=
by sorry

end NUMINAMATH_CALUDE_households_using_only_brand_A_l3858_385834


namespace NUMINAMATH_CALUDE_cube_root_8000_simplification_l3858_385894

theorem cube_root_8000_simplification :
  ∃ (a b : ℕ+), (a : ℝ) * ((b : ℝ) ^ (1/3 : ℝ)) = (8000 : ℝ) ^ (1/3 : ℝ) ∧ 
  (∀ (c d : ℕ+), (c : ℝ) * ((d : ℝ) ^ (1/3 : ℝ)) = (8000 : ℝ) ^ (1/3 : ℝ) → d ≥ b) ∧
  a = 20 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_8000_simplification_l3858_385894


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3858_385861

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) = (a^2 - 3*a + 2) + Complex.I * (a - 1)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3858_385861


namespace NUMINAMATH_CALUDE_circles_shortest_distance_l3858_385847

/-- Definition of the first circle -/
def circle1 (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 6*y = 8

/-- Definition of the second circle -/
def circle2 (x y : ℝ) : Prop :=
  x^2 + 6*x + y^2 - 2*y = 1

/-- The shortest distance between the two circles -/
def shortest_distance : ℝ := -0.68

/-- Theorem stating that the shortest distance between the two circles is -0.68 -/
theorem circles_shortest_distance :
  ∃ (d : ℝ), d = shortest_distance ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    circle1 x₁ y₁ → circle2 x₂ y₂ →
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≥ d :=
  sorry

end NUMINAMATH_CALUDE_circles_shortest_distance_l3858_385847


namespace NUMINAMATH_CALUDE_veronica_yellow_balls_l3858_385822

theorem veronica_yellow_balls :
  let total_balls : ℕ := 60
  let yellow_balls : ℕ := 27
  let brown_balls : ℕ := 33
  (yellow_balls : ℚ) / total_balls = 45 / 100 ∧
  brown_balls + yellow_balls = total_balls →
  yellow_balls = 27 :=
by sorry

end NUMINAMATH_CALUDE_veronica_yellow_balls_l3858_385822


namespace NUMINAMATH_CALUDE_order_cost_l3858_385896

/-- The cost of the order given the prices and quantities of pencils and erasers -/
theorem order_cost (pencil_price eraser_price : ℕ) (total_cartons pencil_cartons : ℕ) : 
  pencil_price = 6 →
  eraser_price = 3 →
  total_cartons = 100 →
  pencil_cartons = 20 →
  pencil_price * pencil_cartons + eraser_price * (total_cartons - pencil_cartons) = 360 := by
sorry

end NUMINAMATH_CALUDE_order_cost_l3858_385896


namespace NUMINAMATH_CALUDE_factorization_eq_l3858_385852

theorem factorization_eq (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_eq_l3858_385852


namespace NUMINAMATH_CALUDE_presentation_length_appropriate_l3858_385840

/-- Represents the duration of a presentation in minutes -/
def PresentationDuration := { d : ℝ // 45 ≤ d ∧ d ≤ 60 }

/-- The recommended speech rate in words per minute -/
def SpeechRate : ℝ := 160

/-- Checks if a given number of words is appropriate for the presentation -/
def isAppropriateLength (duration : PresentationDuration) (words : ℕ) : Prop :=
  (↑words : ℝ) ≥ SpeechRate * duration.val ∧ (↑words : ℝ) ≤ SpeechRate * 60

theorem presentation_length_appropriate :
  ∀ (duration : PresentationDuration), isAppropriateLength duration 9400 := by
  sorry

end NUMINAMATH_CALUDE_presentation_length_appropriate_l3858_385840


namespace NUMINAMATH_CALUDE_power_equation_solution_l3858_385866

theorem power_equation_solution (x y : ℕ+) (h : 2^(x.val + 1) * 4^y.val = 128) : 
  x.val + 2 * y.val = 6 := by
sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3858_385866


namespace NUMINAMATH_CALUDE_ball_falls_in_hole_iff_ratio_rational_l3858_385841

/-- A pool table with sides a and b -/
structure PoolTable where
  a : ℝ
  b : ℝ

/-- Predicate to check if a ratio is rational -/
def isRational (x : ℝ) : Prop :=
  ∃ (m n : ℤ), n ≠ 0 ∧ x = m / n

/-- Predicate to check if a ball shot from a corner along the angle bisector falls into a hole -/
def ballFallsInHole (table : PoolTable) : Prop :=
  ∃ (k l : ℤ), k ≠ 0 ∧ l ≠ 0 ∧ table.a * (1 - k) = table.b * (l - 1)

/-- Theorem stating the condition for a ball to fall into a hole -/
theorem ball_falls_in_hole_iff_ratio_rational (table : PoolTable) :
  ballFallsInHole table ↔ isRational (table.a / table.b) :=
sorry

end NUMINAMATH_CALUDE_ball_falls_in_hole_iff_ratio_rational_l3858_385841


namespace NUMINAMATH_CALUDE_alicia_tax_deduction_l3858_385871

/-- Calculates the local tax deduction in cents given an hourly wage in dollars and a tax rate as a percentage. -/
def localTaxDeduction (hourlyWage : ℚ) (taxRate : ℚ) : ℚ :=
  hourlyWage * 100 * (taxRate / 100)

/-- Theorem stating that for an hourly wage of $25 and a tax rate of 2.2%, the local tax deduction is 55 cents. -/
theorem alicia_tax_deduction :
  localTaxDeduction 25 2.2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_alicia_tax_deduction_l3858_385871


namespace NUMINAMATH_CALUDE_min_garden_width_proof_l3858_385898

/-- The minimum width of a rectangular garden satisfying the given conditions -/
def min_garden_width : ℝ := 4

/-- The length of the garden in terms of its width -/
def garden_length (w : ℝ) : ℝ := w + 20

/-- The area of the garden in terms of its width -/
def garden_area (w : ℝ) : ℝ := w * garden_length w

theorem min_garden_width_proof :
  (∀ w : ℝ, w > 0 → garden_area w ≥ 120 → w ≥ min_garden_width) ∧
  garden_area min_garden_width ≥ 120 :=
sorry

end NUMINAMATH_CALUDE_min_garden_width_proof_l3858_385898


namespace NUMINAMATH_CALUDE_frustum_volume_ratio_l3858_385810

theorem frustum_volume_ratio (h₁ h₂ : ℝ) (A₁ A₂ : ℝ) (V₁ V₂ : ℝ) :
  h₁ / h₂ = 3 / 5 →
  A₁ / A₂ = 9 / 25 →
  V₁ = (1 / 3) * h₁ * A₁ →
  V₂ = (1 / 3) * h₂ * A₂ →
  V₁ / (V₂ - V₁) = 27 / 71 :=
by sorry

end NUMINAMATH_CALUDE_frustum_volume_ratio_l3858_385810


namespace NUMINAMATH_CALUDE_tv_purchase_price_l3858_385893

/-- Proves that the purchase price of a TV is 1200 yuan given the markup, promotion, and profit conditions. -/
theorem tv_purchase_price (x : ℝ) 
  (markup : ℝ → ℝ) 
  (promotion : ℝ → ℝ) 
  (profit : ℝ) 
  (h1 : markup x = 1.35 * x) 
  (h2 : promotion (markup x) = 0.9 * markup x - 50)
  (h3 : promotion (markup x) - x = profit)
  (h4 : profit = 208) : 
  x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_tv_purchase_price_l3858_385893


namespace NUMINAMATH_CALUDE_annulus_area_l3858_385830

/-- The area of an annulus with outer radius 8 feet and inner radius 2 feet is 60π square feet. -/
theorem annulus_area : ∀ (π : ℝ), π > 0 → π * (8^2 - 2^2) = 60 * π := by
  sorry

end NUMINAMATH_CALUDE_annulus_area_l3858_385830


namespace NUMINAMATH_CALUDE_average_balance_is_200_l3858_385857

/-- Represents the balance of a savings account for a given month -/
structure MonthlyBalance where
  month : String
  balance : ℕ

/-- Calculates the average monthly balance given a list of monthly balances -/
def averageMonthlyBalance (balances : List MonthlyBalance) : ℚ :=
  (balances.map (·.balance)).sum / balances.length

/-- Theorem stating that the average monthly balance is $200 -/
theorem average_balance_is_200 (balances : List MonthlyBalance) 
  (h1 : balances = [
    { month := "January", balance := 200 },
    { month := "February", balance := 300 },
    { month := "March", balance := 100 },
    { month := "April", balance := 250 },
    { month := "May", balance := 150 }
  ]) : 
  averageMonthlyBalance balances = 200 := by
  sorry


end NUMINAMATH_CALUDE_average_balance_is_200_l3858_385857


namespace NUMINAMATH_CALUDE_total_outfits_count_l3858_385835

def total_shirts : ℕ := 8
def total_ties : ℕ := 6
def special_shirt_matches : ℕ := 3

def outfits_with_special_shirt : ℕ := special_shirt_matches
def outfits_with_other_shirts : ℕ := (total_shirts - 1) * total_ties

theorem total_outfits_count :
  outfits_with_special_shirt + outfits_with_other_shirts = 45 :=
by sorry

end NUMINAMATH_CALUDE_total_outfits_count_l3858_385835


namespace NUMINAMATH_CALUDE_rectangle_cut_and_rearrange_l3858_385848

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the result of cutting and rearranging a rectangle -/
structure CutAndRearrange where
  original : Rectangle
  new : Rectangle

/-- Defines the properties of a valid cut and rearrange operation -/
def isValidCutAndRearrange (cr : CutAndRearrange) : Prop :=
  cr.original.width * cr.original.height = cr.new.width * cr.new.height ∧
  cr.new.width ≠ cr.original.width ∧
  cr.new.height ≠ cr.original.height ∧
  (cr.new.width > cr.new.height → cr.new.width > cr.original.width ∧ cr.new.width > cr.original.height) ∧
  (cr.new.height > cr.new.width → cr.new.height > cr.original.width ∧ cr.new.height > cr.original.height)

/-- The main theorem to be proved -/
theorem rectangle_cut_and_rearrange :
  ∀ (cr : CutAndRearrange),
    cr.original.width = 9 ∧
    cr.original.height = 16 ∧
    isValidCutAndRearrange cr →
    max cr.new.width cr.new.height = 18 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_cut_and_rearrange_l3858_385848


namespace NUMINAMATH_CALUDE_mean_home_runs_l3858_385802

def player_count : ℕ := 13
def total_home_runs : ℕ := 80

def home_run_distribution : List (ℕ × ℕ) :=
  [(5, 5), (5, 6), (1, 7), (1, 8), (1, 10)]

theorem mean_home_runs :
  (total_home_runs : ℚ) / player_count = 80 / 13 := by
  sorry

end NUMINAMATH_CALUDE_mean_home_runs_l3858_385802


namespace NUMINAMATH_CALUDE_house_development_l3858_385844

theorem house_development (total houses garage pool neither : ℕ) : 
  total = 70 → 
  garage = 50 → 
  pool = 40 → 
  neither = 15 → 
  ∃ both : ℕ, both = garage + pool - (total - neither) :=
by
  sorry

end NUMINAMATH_CALUDE_house_development_l3858_385844


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3858_385860

theorem line_segment_endpoint (x : ℝ) :
  x > 0 →
  (((x - 2)^2 + (6 - 2)^2).sqrt = 7) →
  x = 2 + Real.sqrt 33 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3858_385860


namespace NUMINAMATH_CALUDE_dans_final_limes_l3858_385867

def initial_limes : ℕ := 9
def sara_gift : ℕ := 4
def juice_used : ℕ := 5
def neighbor_gift : ℕ := 3

theorem dans_final_limes : 
  initial_limes + sara_gift - juice_used - neighbor_gift = 5 := by
  sorry

end NUMINAMATH_CALUDE_dans_final_limes_l3858_385867


namespace NUMINAMATH_CALUDE_tetrahedron_sum_l3858_385854

/-- A tetrahedron is a three-dimensional geometric shape with four faces, four vertices, and six edges. --/
structure Tetrahedron where
  edges : Nat
  corners : Nat
  faces : Nat

/-- The sum of edges, corners, and faces of a tetrahedron is 14. --/
theorem tetrahedron_sum (t : Tetrahedron) : t.edges + t.corners + t.faces = 14 := by
  sorry

#check tetrahedron_sum

end NUMINAMATH_CALUDE_tetrahedron_sum_l3858_385854


namespace NUMINAMATH_CALUDE_derivative_cosh_l3858_385876

open Real

theorem derivative_cosh (x : ℝ) : 
  deriv (fun x => (1/2) * (exp x + exp (-x))) x = (1/2) * (exp x - exp (-x)) := by
  sorry

end NUMINAMATH_CALUDE_derivative_cosh_l3858_385876


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3858_385880

theorem sphere_surface_area (d : ℝ) (h : d = 4) :
  4 * Real.pi * (d / 2)^2 = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3858_385880


namespace NUMINAMATH_CALUDE_f_properties_l3858_385886

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem f_properties :
  (∀ x : ℝ, f x ≥ 3 ↔ x ≤ -3/2 ∨ x ≥ 3/2) ∧
  (∀ a : ℝ, (∀ x : ℝ, f x ≥ a^2 - a) ↔ a ∈ Set.Icc (-1) 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3858_385886


namespace NUMINAMATH_CALUDE_muffin_goal_remaining_l3858_385843

def muffin_problem (goal : ℕ) (morning_sales : ℕ) (afternoon_sales : ℕ) : ℕ :=
  goal - (morning_sales + afternoon_sales)

theorem muffin_goal_remaining :
  muffin_problem 20 12 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_muffin_goal_remaining_l3858_385843


namespace NUMINAMATH_CALUDE_problem_solution_l3858_385815

theorem problem_solution (x : ℂ) (h : x + 1/x = -1) : x^1994 + 1/x^1994 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3858_385815


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_values_l3858_385814

-- Define the properties of the function f
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_two_property (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f x

-- State the theorem
theorem sum_of_four_consecutive_values (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_period : has_period_two_property f) : 
  f 2008 + f 2009 + f 2010 + f 2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_values_l3858_385814


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l3858_385842

theorem quadratic_solution_property (a b : ℝ) : 
  (a * 1^2 + b * 1 + 1 = 0) → (3 - a - b = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l3858_385842


namespace NUMINAMATH_CALUDE_square_perimeter_l3858_385833

theorem square_perimeter (s : ℝ) (h1 : s > 0) (h2 : s^2 = 625) : 4 * s = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3858_385833


namespace NUMINAMATH_CALUDE_snack_distribution_solution_l3858_385826

/-- Represents the snack distribution problem for a kindergarten class. -/
structure SnackDistribution where
  pretzels : ℕ
  goldfish : ℕ
  suckers : ℕ
  kids : ℕ
  pretzel_popcorn_ratio : ℚ

/-- Calculates the number of items per snack type in each baggie. -/
def items_per_baggie (sd : SnackDistribution) : ℕ × ℕ × ℕ × ℕ :=
  let pretzels_per_baggie := sd.pretzels / sd.kids
  let goldfish_per_baggie := sd.goldfish / sd.kids
  let suckers_per_baggie := sd.suckers / sd.kids
  let popcorn_per_baggie := (sd.pretzel_popcorn_ratio * pretzels_per_baggie).ceil.toNat
  (pretzels_per_baggie, goldfish_per_baggie, suckers_per_baggie, popcorn_per_baggie)

/-- Calculates the total number of popcorn pieces needed. -/
def total_popcorn (sd : SnackDistribution) : ℕ :=
  let (_, _, _, popcorn_per_baggie) := items_per_baggie sd
  popcorn_per_baggie * sd.kids

/-- Calculates the total number of items in each baggie. -/
def total_items_per_baggie (sd : SnackDistribution) : ℕ :=
  let (p, g, s, c) := items_per_baggie sd
  p + g + s + c

/-- Theorem stating the solution to the snack distribution problem. -/
theorem snack_distribution_solution (sd : SnackDistribution) 
  (h1 : sd.pretzels = 64)
  (h2 : sd.goldfish = 4 * sd.pretzels)
  (h3 : sd.suckers = 32)
  (h4 : sd.kids = 23)
  (h5 : sd.pretzel_popcorn_ratio = 3/2) :
  total_popcorn sd = 69 ∧ total_items_per_baggie sd = 17 := by
  sorry


end NUMINAMATH_CALUDE_snack_distribution_solution_l3858_385826


namespace NUMINAMATH_CALUDE_not_all_odd_have_all_five_multiple_l3858_385816

theorem not_all_odd_have_all_five_multiple : ∃ n : ℕ, Odd n ∧ ∀ k : ℕ, ∃ d : ℕ, d ≠ 5 ∧ d ∈ (k * n).digits 10 := by
  sorry

end NUMINAMATH_CALUDE_not_all_odd_have_all_five_multiple_l3858_385816


namespace NUMINAMATH_CALUDE_xy_value_l3858_385891

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3858_385891


namespace NUMINAMATH_CALUDE_sqrt_square_abs_sqrt_neg_five_squared_l3858_385895

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

theorem sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_sqrt_neg_five_squared_l3858_385895


namespace NUMINAMATH_CALUDE_pair_five_cows_four_pigs_seven_horses_l3858_385851

/-- The number of ways to pair animals of different species -/
def pairAnimals (cows pigs horses : ℕ) : ℕ :=
  cows * pigs * (cows + pigs - 2).factorial

/-- Theorem stating the number of ways to pair 5 cows, 4 pigs, and 7 horses -/
theorem pair_five_cows_four_pigs_seven_horses :
  pairAnimals 5 4 7 = 100800 := by
  sorry

#eval pairAnimals 5 4 7

end NUMINAMATH_CALUDE_pair_five_cows_four_pigs_seven_horses_l3858_385851


namespace NUMINAMATH_CALUDE_nature_reserve_count_l3858_385885

theorem nature_reserve_count (total_heads total_legs : ℕ) 
  (h1 : total_heads = 300)
  (h2 : total_legs = 688) : ∃ (birds mammals reptiles : ℕ),
  birds + mammals + reptiles = total_heads ∧
  2 * birds + 4 * mammals + 6 * reptiles = total_legs ∧
  birds = 234 := by
  sorry

end NUMINAMATH_CALUDE_nature_reserve_count_l3858_385885


namespace NUMINAMATH_CALUDE_arithmetic_progression_condition_l3858_385839

def list : List ℤ := [3, 7, 2, 7, 5, 2]

def mean (x : ℚ) : ℚ := (list.sum + x) / 7

def mode : ℤ := 7

noncomputable def median (x : ℚ) : ℚ :=
  if x ≤ 2 then 3
  else if x < 5 then x
  else 5

theorem arithmetic_progression_condition (x : ℚ) :
  (mode : ℚ) < median x ∧ median x < mean x ∧
  median x - mode = mean x - median x →
  x = 75 / 13 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_condition_l3858_385839


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3858_385821

def A : Set ℝ := {x | x - 1 ≤ 0}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3858_385821


namespace NUMINAMATH_CALUDE_clarence_oranges_l3858_385874

/-- The number of oranges Clarence had initially -/
def initial_oranges : ℕ := sorry

/-- The number of oranges Clarence received from Joyce -/
def oranges_from_joyce : ℕ := 3

/-- The total number of oranges Clarence has after receiving oranges from Joyce -/
def total_oranges : ℕ := 8

/-- Theorem stating that the initial number of oranges plus those from Joyce equals the total -/
theorem clarence_oranges : initial_oranges + oranges_from_joyce = total_oranges := by sorry

end NUMINAMATH_CALUDE_clarence_oranges_l3858_385874
