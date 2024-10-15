import Mathlib

namespace NUMINAMATH_CALUDE_solve_chalk_problem_l3133_313377

def chalk_problem (siblings friends chalk_per_person lost_chalk : ℕ) : Prop :=
  let total_people : ℕ := siblings + friends
  let total_chalk_needed : ℕ := total_people * chalk_per_person
  let available_chalk : ℕ := total_chalk_needed - lost_chalk
  let mom_brought : ℕ := total_chalk_needed - available_chalk
  mom_brought = 2

theorem solve_chalk_problem :
  chalk_problem 4 3 3 2 := by sorry

end NUMINAMATH_CALUDE_solve_chalk_problem_l3133_313377


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_minimum_l3133_313345

theorem geometric_sequence_sum_minimum (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  q > 1 →
  (∀ n, a n = a 1 * q^(n-1)) →
  (∀ n, S n = (a 1 * (1 - q^n)) / (1 - q)) →
  S 4 = 2 * S 2 + 1 →
  ∃ S_6_min : ℝ, S_6_min = 2 * Real.sqrt 3 + 3 ∧ S 6 ≥ S_6_min :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_minimum_l3133_313345


namespace NUMINAMATH_CALUDE_triangle_inequality_l3133_313354

theorem triangle_inequality (a b c p : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : p = (a + b + c) / 2) :
  Real.sqrt (p - a) + Real.sqrt (p - b) + Real.sqrt (p - c) ≤ Real.sqrt (3 * p) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3133_313354


namespace NUMINAMATH_CALUDE_adjacent_even_numbers_l3133_313396

theorem adjacent_even_numbers (n : ℕ) : 
  Odd (2*n + 1) ∧ 
  Even (2*n) ∧ 
  Even (2*n + 2) ∧
  (2*n + 1) - 1 = 2*n ∧
  (2*n + 1) + 1 = 2*n + 2 := by
sorry

end NUMINAMATH_CALUDE_adjacent_even_numbers_l3133_313396


namespace NUMINAMATH_CALUDE_special_ellipse_minor_axis_length_special_ellipse_minor_axis_length_is_four_l3133_313303

/-- An ellipse passing through five given points with specific properties -/
structure SpecialEllipse where
  -- The five points the ellipse passes through
  p₁ : ℝ × ℝ := (-1, -1)
  p₂ : ℝ × ℝ := (0, 0)
  p₃ : ℝ × ℝ := (0, 4)
  p₄ : ℝ × ℝ := (4, 0)
  p₅ : ℝ × ℝ := (4, 4)
  -- The center of the ellipse
  center : ℝ × ℝ := (2, 2)
  -- The ellipse has axes parallel to the coordinate axes
  axes_parallel : Bool

/-- The length of the minor axis of the special ellipse is 4 -/
theorem special_ellipse_minor_axis_length (e : SpecialEllipse) : ℝ :=
  4

/-- The main theorem: The length of the minor axis of the special ellipse is 4 -/
theorem special_ellipse_minor_axis_length_is_four (e : SpecialEllipse) :
  special_ellipse_minor_axis_length e = 4 := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_minor_axis_length_special_ellipse_minor_axis_length_is_four_l3133_313303


namespace NUMINAMATH_CALUDE_number_difference_l3133_313371

theorem number_difference (a b : ℕ) (h1 : a + b = 27630) (h2 : 5 * a + 5 = b) :
  b - a = 18421 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3133_313371


namespace NUMINAMATH_CALUDE_min_cost_for_ten_boxes_l3133_313322

/-- Calculates the minimum cost for buying a given number of yogurt boxes under a "buy two get one free" promotion. -/
def min_cost (box_price : ℕ) (num_boxes : ℕ) : ℕ :=
  let full_price_boxes := (num_boxes + 2) / 3 * 2
  full_price_boxes * box_price

/-- Theorem stating that the minimum cost for 10 boxes of yogurt at 4 yuan each under the promotion is 28 yuan. -/
theorem min_cost_for_ten_boxes : min_cost 4 10 = 28 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_for_ten_boxes_l3133_313322


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3133_313340

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  first_positive : a 1 > 0
  sum_condition : a 1 + a 3 + a 5 = 6
  product_condition : a 1 * a 3 * a 5 = 0
  is_arithmetic : ∀ n m : ℕ+, a (n + m) - a n = m * (a 2 - a 1)

/-- The general term of the sequence -/
def general_term (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  5 - n

/-- The b_n term -/
def b (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  1 / (n * (seq.a n - 6))

/-- The sum of the first n terms of b_n -/
def S (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  -n / (n + 1)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ+, seq.a n = general_term seq n) ∧
  (∀ n : ℕ+, S seq n = -n / (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3133_313340


namespace NUMINAMATH_CALUDE_michelle_initial_crayons_l3133_313318

/-- Given that Janet has 2 crayons and the sum of Michelle's initial crayons
    and Janet's crayons is 4, prove that Michelle initially has 2 crayons. -/
theorem michelle_initial_crayons :
  ∀ (michelle_initial janet : ℕ),
    janet = 2 →
    michelle_initial + janet = 4 →
    michelle_initial = 2 := by
  sorry

end NUMINAMATH_CALUDE_michelle_initial_crayons_l3133_313318


namespace NUMINAMATH_CALUDE_max_value_expression_l3133_313314

theorem max_value_expression (x y : ℝ) : 2 * y^2 - y^4 - x^2 - 3 * x ≤ 13/4 := by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3133_313314


namespace NUMINAMATH_CALUDE_impossible_last_digit_match_l3133_313364

theorem impossible_last_digit_match (n : ℕ) (h_n : n = 111) :
  ¬ ∃ (S : Finset ℕ),
    Finset.card S = n ∧
    (∀ x ∈ S, x ≤ 500) ∧
    (∀ x ∈ S, ∀ y ∈ S, x ≠ y → x ≠ y) ∧
    (∀ x ∈ S, x % 10 = (Finset.sum S id - x) % 10) :=
by sorry

end NUMINAMATH_CALUDE_impossible_last_digit_match_l3133_313364


namespace NUMINAMATH_CALUDE_sqrt_four_equals_plus_minus_two_l3133_313398

theorem sqrt_four_equals_plus_minus_two : ∀ (x : ℝ), x^2 = 4 → x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_equals_plus_minus_two_l3133_313398


namespace NUMINAMATH_CALUDE_root_implies_a_range_l3133_313333

def f (a x : ℝ) : ℝ := 2 * a * x^2 + 2 * x - 3 - a

theorem root_implies_a_range (a : ℝ) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = 0) →
  a ≥ 1 ∨ a ≤ -(3 + Real.sqrt 7) / 2 :=
by sorry

end NUMINAMATH_CALUDE_root_implies_a_range_l3133_313333


namespace NUMINAMATH_CALUDE_sin_beta_value_l3133_313311

theorem sin_beta_value (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos α = 4 / 5)
  (h4 : Real.cos (α + β) = 3 / 5) : 
  Real.sin β = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_beta_value_l3133_313311


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l3133_313395

noncomputable def triangle_side_length (angleA : Real) (sideBC : Real) : Real :=
  sideBC * Real.tan angleA

theorem right_triangle_side_length :
  let angleA : Real := 30 * π / 180  -- Convert 30° to radians
  let sideBC : Real := 12
  let sideAB : Real := triangle_side_length angleA sideBC
  ∀ ε > 0, |sideAB - 6.9| < ε :=
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l3133_313395


namespace NUMINAMATH_CALUDE_fraction_change_l3133_313369

/-- Given a fraction that changes from 1/12 to 2/15 when its numerator is increased by 20% and
    its denominator is decreased by x%, prove that x = 25. -/
theorem fraction_change (x : ℚ) : 
  (1 : ℚ) / 12 * (120 / 100) / ((100 - x) / 100) = 2 / 15 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_change_l3133_313369


namespace NUMINAMATH_CALUDE_solution_comparison_l3133_313304

theorem solution_comparison (a a' b b' c : ℝ) (ha : a ≠ 0) (ha' : a' ≠ 0) :
  ((c - b) / a > (c - b') / a') ↔ ((c - b') / a' < (c - b) / a) := by sorry

end NUMINAMATH_CALUDE_solution_comparison_l3133_313304


namespace NUMINAMATH_CALUDE_taxi_charge_calculation_l3133_313308

/-- Calculates the taxi charge per mile given the initial fee, total distance, and total payment. -/
def taxi_charge_per_mile (initial_fee : ℚ) (total_distance : ℚ) (total_payment : ℚ) : ℚ :=
  (total_payment - initial_fee) / total_distance

/-- Theorem stating that the taxi charge per mile is $2.50 given the specific conditions. -/
theorem taxi_charge_calculation (initial_fee : ℚ) (total_distance : ℚ) (total_payment : ℚ)
    (h1 : initial_fee = 2)
    (h2 : total_distance = 4)
    (h3 : total_payment = 12) :
    taxi_charge_per_mile initial_fee total_distance total_payment = 2.5 := by
  sorry

#eval taxi_charge_per_mile 2 4 12

end NUMINAMATH_CALUDE_taxi_charge_calculation_l3133_313308


namespace NUMINAMATH_CALUDE_amandas_family_size_l3133_313336

theorem amandas_family_size :
  let total_rooms : ℕ := 9
  let rooms_with_four_walls : ℕ := 5
  let rooms_with_five_walls : ℕ := 4
  let walls_per_person : ℕ := 8
  let total_walls : ℕ := rooms_with_four_walls * 4 + rooms_with_five_walls * 5
  total_rooms = rooms_with_four_walls + rooms_with_five_walls →
  total_walls % walls_per_person = 0 →
  total_walls / walls_per_person = 5 :=
by sorry

end NUMINAMATH_CALUDE_amandas_family_size_l3133_313336


namespace NUMINAMATH_CALUDE_adjusted_retail_price_l3133_313315

/-- The adjusted retail price of a shirt given its cost price and price adjustments -/
theorem adjusted_retail_price 
  (a : ℝ) -- Cost price per shirt in yuan
  (m : ℝ) -- Initial markup percentage
  (n : ℝ) -- Price adjustment percentage
  : ℝ := by
  -- The adjusted retail price is a(1+m%)n% yuan
  sorry

#check adjusted_retail_price

end NUMINAMATH_CALUDE_adjusted_retail_price_l3133_313315


namespace NUMINAMATH_CALUDE_min_cubes_is_four_l3133_313384

/-- Represents a cube with two protruding snaps on opposite sides and four receptacle holes --/
structure Cube where
  snaps : Fin 2
  holes : Fin 4

/-- Represents an assembly of cubes --/
def Assembly := List Cube

/-- Checks if an assembly has only receptacle holes visible --/
def Assembly.onlyHolesVisible (a : Assembly) : Prop :=
  sorry

/-- The minimum number of cubes required for a valid assembly --/
def minCubesForValidAssembly : ℕ :=
  sorry

/-- Theorem stating that the minimum number of cubes for a valid assembly is 4 --/
theorem min_cubes_is_four :
  minCubesForValidAssembly = 4 :=
sorry

end NUMINAMATH_CALUDE_min_cubes_is_four_l3133_313384


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3133_313367

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : a 3 + a 6 = 11)
  (h3 : a 5 + a 8 = 39) :
  ∃ d : ℝ, d = 7 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3133_313367


namespace NUMINAMATH_CALUDE_tv_price_change_l3133_313376

theorem tv_price_change (x : ℝ) : 
  (100 - x) * 1.5 = 120 → x = 20 := by sorry

end NUMINAMATH_CALUDE_tv_price_change_l3133_313376


namespace NUMINAMATH_CALUDE_quadratic_factor_condition_l3133_313337

theorem quadratic_factor_condition (a b p q : ℝ) : 
  (∀ x, (x + a) * (x + b) = x^2 + p*x + q) →
  p > 0 →
  q < 0 →
  ((a > 0 ∧ b < 0 ∧ a > -b) ∨ (a < 0 ∧ b > 0 ∧ b > -a)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factor_condition_l3133_313337


namespace NUMINAMATH_CALUDE_digit_equation_solution_l3133_313330

theorem digit_equation_solution (A M C : ℕ) : 
  A ≤ 9 ∧ M ≤ 9 ∧ C ≤ 9 →
  (100 * A + 10 * M + C) * (A + M + C) = 2040 →
  Even (A + M + C) →
  M = 7 :=
by sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l3133_313330


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3133_313399

theorem quadratic_root_value (k : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 2 * x + k = 0 ↔ x = (1 + Complex.I * Real.sqrt 39) / 10 ∨ x = (1 - Complex.I * Real.sqrt 39) / 10) →
  k = 2.15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3133_313399


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3133_313301

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x + 5 > 3 ∧ x > a) ↔ x > -2) → a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3133_313301


namespace NUMINAMATH_CALUDE_function_property_l3133_313382

theorem function_property (f : ℕ → ℕ) :
  (∀ m n : ℕ, (m^2 + f n) ∣ (m * f m + n)) →
  (∀ n : ℕ, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_function_property_l3133_313382


namespace NUMINAMATH_CALUDE_concert_hat_wearers_l3133_313360

theorem concert_hat_wearers (total_attendees : ℕ) 
  (women_fraction : ℚ) (women_hat_percentage : ℚ) (men_hat_percentage : ℚ) :
  total_attendees = 3000 →
  women_fraction = 2/3 →
  women_hat_percentage = 15/100 →
  men_hat_percentage = 12/100 →
  ↑(total_attendees * (women_fraction * women_hat_percentage + 
    (1 - women_fraction) * men_hat_percentage)) = (420 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_concert_hat_wearers_l3133_313360


namespace NUMINAMATH_CALUDE_divisor_power_difference_l3133_313350

theorem divisor_power_difference (k : ℕ) : 
  (15 ^ k : ℕ) ∣ 759325 → 3 ^ k - k ^ 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisor_power_difference_l3133_313350


namespace NUMINAMATH_CALUDE_fun_run_participation_l3133_313300

/-- Fun Run Participation Theorem -/
theorem fun_run_participation (signed_up_last_year : ℕ) (no_show_last_year : ℕ) : 
  signed_up_last_year = 200 →
  no_show_last_year = 40 →
  (signed_up_last_year - no_show_last_year) * 2 = 320 := by
  sorry

#check fun_run_participation

end NUMINAMATH_CALUDE_fun_run_participation_l3133_313300


namespace NUMINAMATH_CALUDE_simon_makes_three_pies_l3133_313347

/-- The number of blueberry pies Simon can make -/
def blueberry_pies (own_berries nearby_berries berries_per_pie : ℕ) : ℕ :=
  (own_berries + nearby_berries) / berries_per_pie

/-- Proof that Simon can make 3 blueberry pies -/
theorem simon_makes_three_pies :
  blueberry_pies 100 200 100 = 3 := by
  sorry

end NUMINAMATH_CALUDE_simon_makes_three_pies_l3133_313347


namespace NUMINAMATH_CALUDE_smallest_staircase_length_l3133_313335

theorem smallest_staircase_length (n : ℕ) : 
  n > 30 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 ∧
  (∀ m : ℕ, m > 30 ∧ m % 6 = 4 ∧ m % 7 = 3 → m ≥ n) → 
  n = 52 := by
sorry

end NUMINAMATH_CALUDE_smallest_staircase_length_l3133_313335


namespace NUMINAMATH_CALUDE_pure_imaginary_product_theorem_l3133_313328

theorem pure_imaginary_product_theorem (z : ℂ) (a : ℝ) : 
  (∃ b : ℝ, z = b * Complex.I) → 
  (3 - Complex.I) * z = a + Complex.I → 
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_theorem_l3133_313328


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3133_313305

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x) ↔ k = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3133_313305


namespace NUMINAMATH_CALUDE_bouquet_combinations_l3133_313394

theorem bouquet_combinations (total : ℕ) (rose_cost : ℕ) (carnation_cost : ℕ) 
  (h_total : total = 50)
  (h_rose : rose_cost = 3)
  (h_carnation : carnation_cost = 2) :
  (∃ (solutions : Finset (ℕ × ℕ)), 
    solutions.card = 9 ∧ 
    ∀ (r c : ℕ), (r, c) ∈ solutions ↔ rose_cost * r + carnation_cost * c = total) :=
sorry

end NUMINAMATH_CALUDE_bouquet_combinations_l3133_313394


namespace NUMINAMATH_CALUDE_set_A_characterization_intersection_A_B_complement_A_union_B_l3133_313387

-- Define the sets A and B
def A : Set ℝ := {x | (x - 3) / (x + 1) > 0}
def B : Set ℝ := {x | x ≤ 4}

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Theorem statements
theorem set_A_characterization : A = {x | x > 3 ∨ x < -1} := by sorry

theorem intersection_A_B : A ∩ B = {x | 3 < x ∧ x ≤ 4} := by sorry

theorem complement_A_union_B : (Set.compl A) ∪ B = {x | x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_set_A_characterization_intersection_A_B_complement_A_union_B_l3133_313387


namespace NUMINAMATH_CALUDE_unique_solution_l3133_313309

/-- Represents the letters used in the triangle puzzle -/
inductive Letter
| A | B | C | D | E | F

/-- Represents the mapping of letters to numbers -/
def LetterMapping := Letter → Fin 6

/-- Checks if a mapping is valid according to the puzzle rules -/
def is_valid_mapping (m : LetterMapping) : Prop :=
  m Letter.A ≠ m Letter.B ∧ m Letter.A ≠ m Letter.C ∧ m Letter.A ≠ m Letter.D ∧ m Letter.A ≠ m Letter.E ∧ m Letter.A ≠ m Letter.F ∧
  m Letter.B ≠ m Letter.C ∧ m Letter.B ≠ m Letter.D ∧ m Letter.B ≠ m Letter.E ∧ m Letter.B ≠ m Letter.F ∧
  m Letter.C ≠ m Letter.D ∧ m Letter.C ≠ m Letter.E ∧ m Letter.C ≠ m Letter.F ∧
  m Letter.D ≠ m Letter.E ∧ m Letter.D ≠ m Letter.F ∧
  m Letter.E ≠ m Letter.F ∧
  (m Letter.B).val + (m Letter.D).val + (m Letter.E).val = 14 ∧
  (m Letter.C).val + (m Letter.E).val + (m Letter.F).val = 12

/-- The unique solution to the puzzle -/
def solution : LetterMapping :=
  fun l => match l with
  | Letter.A => 0
  | Letter.B => 2
  | Letter.C => 1
  | Letter.D => 4
  | Letter.E => 5
  | Letter.F => 3

/-- Theorem stating that the solution is the only valid mapping -/
theorem unique_solution :
  is_valid_mapping solution ∧ ∀ m : LetterMapping, is_valid_mapping m → m = solution := by
  sorry


end NUMINAMATH_CALUDE_unique_solution_l3133_313309


namespace NUMINAMATH_CALUDE_tan_product_identity_l3133_313383

theorem tan_product_identity : (1 + Real.tan (18 * π / 180)) * (1 + Real.tan (27 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_identity_l3133_313383


namespace NUMINAMATH_CALUDE_largest_value_in_special_sequence_l3133_313349

/-- A sequence of 8 increasing real numbers -/
def IncreasingSequence (a : Fin 8 → ℝ) : Prop :=
  ∀ i j, i < j → a i < a j

/-- Checks if a sequence of 4 numbers is an arithmetic progression with a given common difference -/
def IsArithmeticProgression (a : Fin 4 → ℝ) (d : ℝ) : Prop :=
  ∀ i : Fin 3, a (i + 1) - a i = d

/-- Checks if a sequence of 4 numbers is a geometric progression -/
def IsGeometricProgression (a : Fin 4 → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ i : Fin 3, a (i + 1) / a i = r

/-- The main theorem -/
theorem largest_value_in_special_sequence (a : Fin 8 → ℝ)
  (h_increasing : IncreasingSequence a)
  (h_arithmetic1 : ∃ i : Fin 5, IsArithmeticProgression (fun j => a (i + j)) 4)
  (h_arithmetic2 : ∃ i : Fin 5, IsArithmeticProgression (fun j => a (i + j)) 36)
  (h_geometric : ∃ i : Fin 5, IsGeometricProgression (fun j => a (i + j))) :
  a 7 = 126 ∨ a 7 = 6 :=
sorry

end NUMINAMATH_CALUDE_largest_value_in_special_sequence_l3133_313349


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l3133_313302

theorem average_age_after_leaving (initial_people : ℕ) (initial_avg : ℚ) 
  (leaving_age : ℕ) (remaining_people : ℕ) :
  initial_people = 8 →
  initial_avg = 27 →
  leaving_age = 21 →
  remaining_people = 7 →
  (initial_people : ℚ) * initial_avg - leaving_age ≥ 0 →
  (((initial_people : ℚ) * initial_avg - leaving_age) / remaining_people : ℚ) = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_average_age_after_leaving_l3133_313302


namespace NUMINAMATH_CALUDE_grid_paths_path_count_l3133_313357

theorem grid_paths (n m : ℕ) : (n + m).choose n = (n + m).choose m := by sorry

theorem path_count : Nat.choose 9 4 = 126 := by sorry

end NUMINAMATH_CALUDE_grid_paths_path_count_l3133_313357


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_range_l3133_313344

/-- A piecewise function f dependent on parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 2)*x + 6*a - 1 else a^x

/-- Theorem stating the range of a for which f is monotonically decreasing -/
theorem f_monotone_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) → 3/8 ≤ a ∧ a < 2/3 :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_range_l3133_313344


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l3133_313392

open Set Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x, (x - 1) * (deriv f x - f x) > 0)
variable (h3 : ∀ x, f (2 - x) = f x * exp (2 - 2*x))

-- Define the solution set
def solution_set := {x : ℝ | exp 2 * f (log x) < x * f 2}

-- State the theorem
theorem solution_set_is_open_interval :
  solution_set f = Ioo 1 (exp 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l3133_313392


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3133_313326

/-- A parabola is defined by its equation y² = x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  is_parabola : equation = fun y x => y^2 = x

/-- The distance between the focus and directrix of a parabola -/
def focus_directrix_distance (p : Parabola) : ℝ := sorry

/-- Theorem: The distance between the focus and directrix of the parabola y² = x is 0.5 -/
theorem parabola_focus_directrix_distance :
  ∀ p : Parabola, p.equation = fun y x => y^2 = x → focus_directrix_distance p = 0.5 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3133_313326


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3133_313390

/-- Given an isosceles right triangle with squares on its sides, 
    prove that its area is 32 square units. -/
theorem isosceles_right_triangle_area 
  (a b c : ℝ) 
  (h_isosceles : a = b) 
  (h_right : a^2 + b^2 = c^2) 
  (h_leg_square : a^2 = 64) 
  (h_hypotenuse_square : c^2 = 256) : 
  (1/2) * a^2 = 32 := by
  sorry

#check isosceles_right_triangle_area

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3133_313390


namespace NUMINAMATH_CALUDE_volume_of_rotated_composite_shape_l3133_313339

/-- The volume of a solid formed by rotating a composite shape about the x-axis -/
theorem volume_of_rotated_composite_shape (π : ℝ) :
  let lower_rectangle_height : ℝ := 4
  let lower_rectangle_width : ℝ := 1
  let upper_rectangle_height : ℝ := 1
  let upper_rectangle_width : ℝ := 5
  let volume_lower := π * lower_rectangle_height^2 * lower_rectangle_width
  let volume_upper := π * upper_rectangle_height^2 * upper_rectangle_width
  volume_lower + volume_upper = 21 * π := by
  sorry

end NUMINAMATH_CALUDE_volume_of_rotated_composite_shape_l3133_313339


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3133_313380

theorem fraction_to_decimal : (47 : ℚ) / 160 = 0.29375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3133_313380


namespace NUMINAMATH_CALUDE_inequality_proof_l3133_313370

theorem inequality_proof (f : ℝ → ℝ) (a m n : ℝ) :
  (∀ x, f x = |x + a|) →
  (Set.Icc (-9 : ℝ) 1 = {x | f x ≤ 5}) →
  (m > 0) →
  (n > 0) →
  (1/m + 1/(2*n) = a) →
  m + 2*n ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3133_313370


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3133_313327

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3133_313327


namespace NUMINAMATH_CALUDE_sum_squared_l3133_313362

theorem sum_squared (x y : ℝ) (h1 : x * (x + y) = 30) (h2 : y * (x + y) = 60) :
  (x + y)^2 = 90 := by
sorry

end NUMINAMATH_CALUDE_sum_squared_l3133_313362


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l3133_313317

theorem modular_arithmetic_problem :
  (3 * (7⁻¹ : ZMod 97) + 5 * (13⁻¹ : ZMod 97)) = (73 : ZMod 97) := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l3133_313317


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_sequence_l3133_313321

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_divisibility_sequence (m n : ℕ) (h : m > 0) (h' : n > 0) :
  m ∣ n → fib m ∣ fib n := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_sequence_l3133_313321


namespace NUMINAMATH_CALUDE_ashley_stair_climbing_time_l3133_313393

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem ashley_stair_climbing_time :
  arithmetic_sequence_sum 30 10 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_ashley_stair_climbing_time_l3133_313393


namespace NUMINAMATH_CALUDE_triangle_equilateral_from_sequences_l3133_313343

/-- A triangle with sides a, b, c opposite to angles A, B, C respectively is equilateral
if its angles form an arithmetic sequence and its sides form a geometric sequence. -/
theorem triangle_equilateral_from_sequences (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- positive angles
  A + B + C = π →  -- sum of angles in a triangle
  2 * B = A + C →  -- angles form arithmetic sequence
  b^2 = a * c →  -- sides form geometric sequence
  A = B ∧ B = C ∧ a = b ∧ b = c :=
by sorry

end NUMINAMATH_CALUDE_triangle_equilateral_from_sequences_l3133_313343


namespace NUMINAMATH_CALUDE_gcd_product_l3133_313319

theorem gcd_product (a b n : ℕ) (ha : Nat.gcd a n = 1) (hb : Nat.gcd b n = 1) : 
  Nat.gcd (a * b) n = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_product_l3133_313319


namespace NUMINAMATH_CALUDE_not_function_B_but_others_are_l3133_313338

-- Define the concept of a function
def is_function (f : ℝ → Set ℝ) : Prop :=
  ∀ x : ℝ, ∃! y : ℝ, y ∈ f x

-- Define the relationships
def rel_A (x : ℝ) : Set ℝ := {y | y = 1 / x}
def rel_B (x : ℝ) : Set ℝ := {y | |y| = 2 * x}
def rel_C (x : ℝ) : Set ℝ := {y | y = 2 * x^2}
def rel_D (x : ℝ) : Set ℝ := {y | y = 3 * x^3}

-- Theorem statement
theorem not_function_B_but_others_are :
  (¬ is_function rel_B) ∧ 
  (is_function rel_A) ∧ 
  (is_function rel_C) ∧ 
  (is_function rel_D) :=
sorry

end NUMINAMATH_CALUDE_not_function_B_but_others_are_l3133_313338


namespace NUMINAMATH_CALUDE_r_profit_share_l3133_313378

/-- Represents the profit share of a partner in a business partnership --/
def ProfitShare (initial_ratio : ℚ) (months_full : ℕ) (months_reduced : ℕ) (reduction_factor : ℚ) (total_profit : ℚ) : ℚ :=
  let total_investment := 12 * (4 + 6 + 10)
  let partner_investment := initial_ratio * months_full + initial_ratio * reduction_factor * months_reduced
  (partner_investment / total_investment) * total_profit

theorem r_profit_share :
  let p_ratio : ℚ := 4
  let q_ratio : ℚ := 6
  let r_ratio : ℚ := 10
  let total_profit : ℚ := 4650
  ProfitShare r_ratio 12 0 1 total_profit = 2325 := by
  sorry

end NUMINAMATH_CALUDE_r_profit_share_l3133_313378


namespace NUMINAMATH_CALUDE_job_completion_time_l3133_313334

/-- Given that:
  * A can do a job in 15 days
  * A and B working together for 4 days complete 0.4666666666666667 of the job
  Prove that B can do the job alone in 20 days -/
theorem job_completion_time (a_time : ℝ) (together_time : ℝ) (together_completion : ℝ) (b_time : ℝ) :
  a_time = 15 →
  together_time = 4 →
  together_completion = 0.4666666666666667 →
  together_completion = together_time * (1 / a_time + 1 / b_time) →
  b_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l3133_313334


namespace NUMINAMATH_CALUDE_comic_book_collections_l3133_313385

/-- Kymbrea's initial comic book collection size -/
def kymbrea_initial : ℕ := 50

/-- Kymbrea's monthly comic book addition rate -/
def kymbrea_rate : ℕ := 3

/-- LaShawn's initial comic book collection size -/
def lashawn_initial : ℕ := 20

/-- LaShawn's monthly comic book addition rate -/
def lashawn_rate : ℕ := 8

/-- The number of months after which LaShawn's collection will be three times Kymbrea's -/
def months : ℕ := 130

theorem comic_book_collections :
  lashawn_initial + lashawn_rate * months = 3 * (kymbrea_initial + kymbrea_rate * months) :=
by sorry

end NUMINAMATH_CALUDE_comic_book_collections_l3133_313385


namespace NUMINAMATH_CALUDE_parallelogram_circumference_l3133_313358

/-- The circumference of a parallelogram with side lengths 18 and 12 is 60. -/
theorem parallelogram_circumference : ℝ → ℝ → ℝ → Prop :=
  fun a b c => (a = 18 ∧ b = 12) → c = 2 * (a + b) → c = 60

/-- Proof of the theorem -/
lemma prove_parallelogram_circumference : parallelogram_circumference 18 12 60 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_circumference_l3133_313358


namespace NUMINAMATH_CALUDE_max_roses_for_budget_l3133_313356

/-- Represents the different rose purchasing options --/
inductive RoseOption
  | Individual
  | OneDozen
  | TwoDozen
  | Bulk

/-- Returns the cost of a given rose option --/
def cost (option : RoseOption) : Rat :=
  match option with
  | RoseOption.Individual => 730/100
  | RoseOption.OneDozen => 36
  | RoseOption.TwoDozen => 50
  | RoseOption.Bulk => 200

/-- Returns the number of roses for a given option --/
def roses (option : RoseOption) : Nat :=
  match option with
  | RoseOption.Individual => 1
  | RoseOption.OneDozen => 12
  | RoseOption.TwoDozen => 24
  | RoseOption.Bulk => 100

/-- Represents a purchase of roses --/
structure Purchase where
  individual : Nat
  oneDozen : Nat
  twoDozen : Nat
  bulk : Nat

/-- Calculates the total cost of a purchase --/
def totalCost (p : Purchase) : Rat :=
  p.individual * cost RoseOption.Individual +
  p.oneDozen * cost RoseOption.OneDozen +
  p.twoDozen * cost RoseOption.TwoDozen +
  p.bulk * cost RoseOption.Bulk

/-- Calculates the total number of roses in a purchase --/
def totalRoses (p : Purchase) : Nat :=
  p.individual * roses RoseOption.Individual +
  p.oneDozen * roses RoseOption.OneDozen +
  p.twoDozen * roses RoseOption.TwoDozen +
  p.bulk * roses RoseOption.Bulk

/-- The budget constraint --/
def budget : Rat := 680

/-- Theorem: The maximum number of roses that can be purchased for $680 is 328 --/
theorem max_roses_for_budget :
  ∃ (p : Purchase),
    totalCost p ≤ budget ∧
    totalRoses p = 328 ∧
    ∀ (q : Purchase), totalCost q ≤ budget → totalRoses q ≤ totalRoses p :=
by sorry


end NUMINAMATH_CALUDE_max_roses_for_budget_l3133_313356


namespace NUMINAMATH_CALUDE_sqrt_a_power_b_equals_three_l3133_313312

theorem sqrt_a_power_b_equals_three (a b : ℝ) 
  (h : a^2 - 6*a + Real.sqrt (2*b - 4) = -9) : 
  Real.sqrt (a^b) = 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_a_power_b_equals_three_l3133_313312


namespace NUMINAMATH_CALUDE_value_of_y_l3133_313375

theorem value_of_y (x y : ℝ) (h1 : x^(3*y) = 9) (h2 : x = 3) : y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l3133_313375


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l3133_313346

theorem quadratic_rewrite_sum (a b c : ℝ) :
  (∀ x, 6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) →
  a + b + c = 171 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l3133_313346


namespace NUMINAMATH_CALUDE_sum_always_four_digits_l3133_313368

-- Define nonzero digits
def NonzeroDigit := { n : ℕ | 1 ≤ n ∧ n ≤ 9 }

-- Define the sum function
def sum_numbers (C D : NonzeroDigit) : ℕ :=
  3654 + (100 * C.val + 41) + (10 * D.val + 2) + 111

-- Theorem statement
theorem sum_always_four_digits (C D : NonzeroDigit) :
  ∃ n : ℕ, 1000 ≤ sum_numbers C D ∧ sum_numbers C D < 10000 := by
  sorry

end NUMINAMATH_CALUDE_sum_always_four_digits_l3133_313368


namespace NUMINAMATH_CALUDE_calculation_proof_l3133_313397

theorem calculation_proof : (1 / 6 : ℚ) * (-6 : ℚ) / (-1 / 6 : ℚ) * 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3133_313397


namespace NUMINAMATH_CALUDE_class_average_height_l3133_313342

def average_height_problem (total_students : ℕ) (group1_count : ℕ) (group1_avg : ℝ) (group2_avg : ℝ) : Prop :=
  let group2_count : ℕ := total_students - group1_count
  let total_height : ℝ := group1_count * group1_avg + group2_count * group2_avg
  let class_avg : ℝ := total_height / total_students
  class_avg = 168.6

theorem class_average_height :
  average_height_problem 50 40 169 167 := by
  sorry

end NUMINAMATH_CALUDE_class_average_height_l3133_313342


namespace NUMINAMATH_CALUDE_fourth_vertex_location_l3133_313307

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle defined by its four vertices -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Theorem: Given a rectangle ABCD with specific constraints, the fourth vertex C is at (x, -y) -/
theorem fourth_vertex_location (a y d : ℝ) :
  let O : Point := ⟨0, 0⟩
  let A : Point := ⟨a, 0⟩
  let B : Point := ⟨a, y⟩
  let D : Point := ⟨0, d⟩
  ∀ (rect : Rectangle),
    rect.A = A →
    rect.B = B →
    rect.D = D →
    (O.x - rect.C.x) * (A.x - B.x) + (O.y - rect.C.y) * (A.y - B.y) = 0 →
    (O.x - D.x) * (A.x - rect.C.x) + (O.y - D.y) * (A.y - rect.C.y) = 0 →
    rect.C = ⟨a, -y⟩ :=
by
  sorry


end NUMINAMATH_CALUDE_fourth_vertex_location_l3133_313307


namespace NUMINAMATH_CALUDE_class_size_l3133_313352

/-- The number of students in a class, given information about their sports participation -/
theorem class_size (football : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : football = 26)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 11) :
  football + tennis - both + neither = 40 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l3133_313352


namespace NUMINAMATH_CALUDE_black_cards_taken_out_l3133_313324

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (black_cards : Nat)
  (remaining_black : Nat)

/-- Definition of a standard deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    black_cards := 26,
    remaining_black := 22 }

/-- Theorem: The number of black cards taken out is 4 -/
theorem black_cards_taken_out (d : Deck) (h1 : d = standard_deck) :
  d.black_cards - d.remaining_black = 4 := by
  sorry

end NUMINAMATH_CALUDE_black_cards_taken_out_l3133_313324


namespace NUMINAMATH_CALUDE_sector_central_angle_l3133_313389

theorem sector_central_angle (area : ℝ) (perimeter : ℝ) (r : ℝ) (l : ℝ) (α : ℝ) :
  area = 1 →
  perimeter = 4 →
  2 * r + l = perimeter →
  area = 1/2 * l * r →
  α = l / r →
  α = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3133_313389


namespace NUMINAMATH_CALUDE_max_value_implies_ratio_l3133_313316

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

theorem max_value_implies_ratio (a b : ℝ) :
  (∀ x, f a b x ≤ f a b 1) ∧
  (f a b 1 = 10) →
  a / b = -2/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_ratio_l3133_313316


namespace NUMINAMATH_CALUDE_emerson_rowing_distance_l3133_313351

/-- The total distance covered by Emerson on his rowing trip -/
def total_distance (initial_distance : ℕ) (second_segment : ℕ) (final_segment : ℕ) : ℕ :=
  initial_distance + second_segment + final_segment

/-- Theorem stating that Emerson's total distance is 39 miles -/
theorem emerson_rowing_distance :
  total_distance 6 15 18 = 39 := by
  sorry

end NUMINAMATH_CALUDE_emerson_rowing_distance_l3133_313351


namespace NUMINAMATH_CALUDE_equation_condition_l3133_313313

theorem equation_condition (x y z : ℤ) :
  x * (x - y) + y * (y - z) + z * (z - x) = 0 → x = y ∧ y = z := by
  sorry

end NUMINAMATH_CALUDE_equation_condition_l3133_313313


namespace NUMINAMATH_CALUDE_song_ratio_is_two_to_one_l3133_313310

/-- Represents the number of songs on Aisha's mp3 player at different stages --/
structure SongCount where
  initial : ℕ
  afterWeek : ℕ
  added : ℕ
  removed : ℕ
  final : ℕ

/-- Calculates the ratio of added songs to songs after the first two weeks --/
def songRatio (s : SongCount) : ℚ :=
  s.added / (s.initial + s.afterWeek)

/-- Theorem stating the ratio of added songs to songs after the first two weeks --/
theorem song_ratio_is_two_to_one (s : SongCount)
  (h1 : s.initial = 500)
  (h2 : s.afterWeek = 500)
  (h3 : s.removed = 50)
  (h4 : s.final = 2950)
  (h5 : s.initial + s.afterWeek + s.added - s.removed = s.final) :
  songRatio s = 2 := by
  sorry

#check song_ratio_is_two_to_one

end NUMINAMATH_CALUDE_song_ratio_is_two_to_one_l3133_313310


namespace NUMINAMATH_CALUDE_nested_root_evaluation_l3133_313353

theorem nested_root_evaluation (N : ℝ) (h : N > 1) :
  (N * (N * (N ^ (1/3)) ^ (1/4))) ^ (1/3) = N ^ (4/9) := by
  sorry

end NUMINAMATH_CALUDE_nested_root_evaluation_l3133_313353


namespace NUMINAMATH_CALUDE_most_stable_athlete_l3133_313363

def athlete_variance (a b c d : ℝ) : Prop :=
  a = 0.5 ∧ b = 0.5 ∧ c = 0.6 ∧ d = 0.4

theorem most_stable_athlete (a b c d : ℝ) 
  (h : athlete_variance a b c d) : 
  d < a ∧ d < b ∧ d < c :=
by
  sorry

#check most_stable_athlete

end NUMINAMATH_CALUDE_most_stable_athlete_l3133_313363


namespace NUMINAMATH_CALUDE_parabola_decreasing_implies_m_bound_l3133_313348

/-- If the function y = -x^2 - 4mx + 1 is decreasing on the interval [2, +∞), then m ≥ -1. -/
theorem parabola_decreasing_implies_m_bound (m : ℝ) : 
  (∀ x ≥ 2, ∀ y > x, -y^2 - 4*m*y + 1 < -x^2 - 4*m*x + 1) → 
  m ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_decreasing_implies_m_bound_l3133_313348


namespace NUMINAMATH_CALUDE_solve_star_equation_l3133_313366

-- Define the ★ operation
def star (a b : ℝ) : ℝ := 3 * a - 2 * b^2

-- State the theorem
theorem solve_star_equation : 
  ∃ (a : ℝ), star a 3 = 15 ∧ a = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_star_equation_l3133_313366


namespace NUMINAMATH_CALUDE_percent_relation_l3133_313373

theorem percent_relation (x y : ℝ) (h : 0.6 * (x - y) = 0.3 * (x + y)) : y = (1/3) * x := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3133_313373


namespace NUMINAMATH_CALUDE_min_sum_perfect_squares_l3133_313388

theorem min_sum_perfect_squares (x y : ℤ) (h : x^2 - y^2 = 221) :
  ∃ (a b : ℤ), a^2 - b^2 = 221 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 229 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_perfect_squares_l3133_313388


namespace NUMINAMATH_CALUDE_binomial_10_2_l3133_313386

theorem binomial_10_2 : Nat.choose 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_2_l3133_313386


namespace NUMINAMATH_CALUDE_expression_evaluation_l3133_313355

-- Define the expression
def expression : ℚ := -(2^3) + 6/5 * (2/5)

-- Theorem stating the equality
theorem expression_evaluation : expression = -7 - 13/25 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3133_313355


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3133_313372

/-- Given a train of length 120 m crossing a bridge of length 255 m in 30 seconds,
    prove that the speed of the train is 45 km/hr. -/
theorem train_speed_calculation (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  bridge_length = 255 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l3133_313372


namespace NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l3133_313374

theorem mans_age_to_sons_age_ratio : 
  ∀ (sons_current_age mans_current_age : ℕ),
    sons_current_age = 26 →
    mans_current_age = sons_current_age + 28 →
    (mans_current_age + 2) / (sons_current_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l3133_313374


namespace NUMINAMATH_CALUDE_gold_silver_price_ratio_l3133_313329

/-- Proves that the ratio of gold price to silver price is 50 given the problem conditions --/
theorem gold_silver_price_ratio :
  let silver_amount : ℝ := 1.5
  let gold_amount : ℝ := 2 * silver_amount
  let silver_price : ℝ := 20
  let total_spent : ℝ := 3030
  let gold_price : ℝ := (total_spent - silver_amount * silver_price) / gold_amount
  gold_price / silver_price = 50 := by sorry

end NUMINAMATH_CALUDE_gold_silver_price_ratio_l3133_313329


namespace NUMINAMATH_CALUDE_max_inequality_sqrt_sum_l3133_313391

theorem max_inequality_sqrt_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) :
  ∃ (m : ℝ), m = 2 + Real.sqrt 5 ∧ 
  (∀ (x : ℝ), Real.sqrt (4*a + 1) + Real.sqrt (4*b + 1) + Real.sqrt (4*c + 1) > x → x ≤ m) ∧
  Real.sqrt (4*a + 1) + Real.sqrt (4*b + 1) + Real.sqrt (4*c + 1) > m :=
by sorry

end NUMINAMATH_CALUDE_max_inequality_sqrt_sum_l3133_313391


namespace NUMINAMATH_CALUDE_clock_angle_at_8_clock_angle_at_8_is_120_l3133_313379

/-- The angle between clock hands at 8:00 -/
theorem clock_angle_at_8 : ℝ :=
  let total_degrees : ℝ := 360
  let hours_on_clock : ℕ := 12
  let current_hour : ℕ := 8
  let degrees_per_hour : ℝ := total_degrees / hours_on_clock
  let hour_hand_angle : ℝ := degrees_per_hour * current_hour
  let minute_hand_angle : ℝ := 0
  let angle_diff : ℝ := |hour_hand_angle - minute_hand_angle|
  min angle_diff (total_degrees - angle_diff)

/-- Theorem: The smaller angle between the hour-hand and minute-hand of a clock at 8:00 is 120° -/
theorem clock_angle_at_8_is_120 : clock_angle_at_8 = 120 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_8_clock_angle_at_8_is_120_l3133_313379


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3133_313381

/-- A quadratic function f(x) = x^2 - 3x + m + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 3*x + m + 2

/-- The discriminant of the quadratic function f -/
def discriminant (m : ℝ) : ℝ := (-3)^2 - 4*(1)*(m+2)

theorem quadratic_one_root (m : ℝ) : 
  (∃! x, f m x = 0) → m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3133_313381


namespace NUMINAMATH_CALUDE_computer_discount_theorem_l3133_313325

theorem computer_discount_theorem (saved : ℝ) (paid : ℝ) (additional_discount : ℝ) :
  saved = 120 →
  paid = 1080 →
  additional_discount = 0.05 →
  let original_price := saved + paid
  let first_discount_percentage := saved / original_price
  let second_discount_amount := additional_discount * paid
  let total_saved := saved + second_discount_amount
  let total_percentage_saved := total_saved / original_price
  total_percentage_saved = 0.145 := by
  sorry

end NUMINAMATH_CALUDE_computer_discount_theorem_l3133_313325


namespace NUMINAMATH_CALUDE_special_collection_loans_l3133_313306

theorem special_collection_loans (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : return_rate = 4/5)
  (h3 : final_books = 67) :
  (initial_books - final_books : ℚ) / (1 - return_rate) = 40 := by
  sorry

end NUMINAMATH_CALUDE_special_collection_loans_l3133_313306


namespace NUMINAMATH_CALUDE_a_99_value_l3133_313320

def is_increasing (s : ℕ → ℝ) := ∀ n, s n ≤ s (n + 1)
def is_decreasing (s : ℕ → ℝ) := ∀ n, s n ≥ s (n + 1)

theorem a_99_value (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n ≥ 2, |a n - a (n-1)| = (n : ℝ)^2)
  (h3 : is_increasing (λ n => a (2*n - 1)))
  (h4 : is_decreasing (λ n => a (2*n)))
  (h5 : a 1 > a 2) :
  a 99 = 4950 := by sorry

end NUMINAMATH_CALUDE_a_99_value_l3133_313320


namespace NUMINAMATH_CALUDE_problem_statement_l3133_313331

theorem problem_statement (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 1) : 
  m = 10 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3133_313331


namespace NUMINAMATH_CALUDE_rajesh_savings_l3133_313341

def monthly_salary : ℕ := 15000
def food_percentage : ℚ := 40 / 100
def medicine_percentage : ℚ := 20 / 100
def savings_percentage : ℚ := 60 / 100

theorem rajesh_savings : 
  let remaining := monthly_salary - (monthly_salary * food_percentage + monthly_salary * medicine_percentage)
  ↑(remaining * savings_percentage) = 3600 := by sorry

end NUMINAMATH_CALUDE_rajesh_savings_l3133_313341


namespace NUMINAMATH_CALUDE_area_PQR_l3133_313361

/-- Triangle ABC with given side lengths and points M, N, P, Q, R as described -/
structure TriangleABC where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Point M on AB
  AM : ℝ
  MB : ℝ
  -- Point N on BC
  CN : ℝ
  NB : ℝ
  -- Conditions
  side_lengths : AB = 20 ∧ BC = 21 ∧ CA = 29
  M_ratio : AM / MB = 3 / 2
  N_ratio : CN / NB = 2
  -- Existence of points P, Q, R (not explicitly defined)
  P_exists : ∃ P : ℝ × ℝ, True  -- P on AC
  Q_exists : ∃ Q : ℝ × ℝ, True  -- Q on AC
  R_exists : ∃ R : ℝ × ℝ, True  -- R as intersection of MP and NQ
  MP_parallel_BC : True  -- MP is parallel to BC
  NQ_parallel_AB : True  -- NQ is parallel to AB

/-- The area of triangle PQR is 224/15 -/
theorem area_PQR (t : TriangleABC) : ∃ area_PQR : ℝ, area_PQR = 224/15 := by
  sorry

end NUMINAMATH_CALUDE_area_PQR_l3133_313361


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3133_313332

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 3, 4, 5}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3133_313332


namespace NUMINAMATH_CALUDE_iodine_atom_radius_scientific_notation_l3133_313365

theorem iodine_atom_radius_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.0000000133 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ n = -8 :=
by sorry

end NUMINAMATH_CALUDE_iodine_atom_radius_scientific_notation_l3133_313365


namespace NUMINAMATH_CALUDE_art_department_probabilities_l3133_313359

/-- The number of male members in the student art department -/
def num_males : ℕ := 4

/-- The number of female members in the student art department -/
def num_females : ℕ := 3

/-- The total number of members in the student art department -/
def total_members : ℕ := num_males + num_females

/-- The number of members to be selected for the art performance event -/
def num_selected : ℕ := 2

/-- The probability of selecting exactly one female member -/
def prob_one_female : ℚ := 4 / 7

/-- The probability of selecting a specific female member given a specific male member is selected -/
def prob_female_given_male : ℚ := 1 / 6

theorem art_department_probabilities :
  (prob_one_female = 4 / 7) ∧
  (prob_female_given_male = 1 / 6) := by
  sorry

#check art_department_probabilities

end NUMINAMATH_CALUDE_art_department_probabilities_l3133_313359


namespace NUMINAMATH_CALUDE_acorn_price_multiple_l3133_313323

theorem acorn_price_multiple :
  let alice_acorns : ℕ := 3600
  let alice_price_per_acorn : ℕ := 15
  let bob_total_payment : ℕ := 6000
  let alice_total_payment := alice_acorns * alice_price_per_acorn
  (alice_total_payment : ℚ) / bob_total_payment = 9 := by
  sorry

end NUMINAMATH_CALUDE_acorn_price_multiple_l3133_313323
