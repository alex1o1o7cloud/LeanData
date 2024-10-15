import Mathlib

namespace NUMINAMATH_CALUDE_union_of_sets_l3923_392330

def set_A : Set ℝ := {x | x * (x + 1) ≤ 0}
def set_B : Set ℝ := {x | -1 < x ∧ x < 1}

theorem union_of_sets : set_A ∪ set_B = {x | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l3923_392330


namespace NUMINAMATH_CALUDE_chad_earnings_problem_l3923_392353

/-- Chad's earnings and savings problem -/
theorem chad_earnings_problem (mowing_earnings : ℝ) : 
  (mowing_earnings + 250 + 150 + 150) * 0.4 = 460 → mowing_earnings = 600 := by
  sorry

end NUMINAMATH_CALUDE_chad_earnings_problem_l3923_392353


namespace NUMINAMATH_CALUDE_lockridge_marching_band_max_size_l3923_392377

theorem lockridge_marching_band_max_size :
  ∀ n : ℕ,
  (22 * n ≡ 2 [ZMOD 24]) →
  (22 * n < 1000) →
  (∀ m : ℕ, (22 * m ≡ 2 [ZMOD 24]) → (22 * m < 1000) → (22 * m ≤ 22 * n)) →
  22 * n = 770 :=
by sorry

end NUMINAMATH_CALUDE_lockridge_marching_band_max_size_l3923_392377


namespace NUMINAMATH_CALUDE_tangent_line_and_minimum_l3923_392331

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x - 1

theorem tangent_line_and_minimum (a : ℝ) :
  (∃ y, x - 4 * y + 4 * Real.log 2 - 4 = 0 ↔ 
    y = f 1 x ∧ x = 2) ∧
  (a ≤ 0 → ∀ x ∈ Set.Ioo 0 (Real.exp 1), ∃ y ∈ Set.Ioo 0 (Real.exp 1), f a y < f a x) ∧
  (0 < a → a < Real.exp 1 → ∀ x ∈ Set.Ioo 0 (Real.exp 1), f a a ≤ f a x ∧ f a a = Real.log a) ∧
  (Real.exp 1 ≤ a → ∀ x ∈ Set.Ioo 0 (Real.exp 1), a / Real.exp 1 ≤ f a x ∧ a / Real.exp 1 = f a (Real.exp 1)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_minimum_l3923_392331


namespace NUMINAMATH_CALUDE_library_books_count_l3923_392397

theorem library_books_count :
  ∀ (total_books : ℕ),
    (total_books : ℝ) * 0.8 * 0.4 = 736 →
    total_books = 2300 :=
by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l3923_392397


namespace NUMINAMATH_CALUDE_min_value_of_function_l3923_392322

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  ∃ (m : ℝ), m = 8 ∧ ∀ y, y = x + 1/x + 16*x/(x^2 + 1) → y ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3923_392322


namespace NUMINAMATH_CALUDE_area_of_region_t_l3923_392304

/-- A rhombus with side length 3 and one right angle -/
structure RightRhombus where
  side_length : ℝ
  angle_q : ℝ
  side_length_eq : side_length = 3
  angle_q_eq : angle_q = 90

/-- The region T inside the rhombus -/
def region_t (r : RightRhombus) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the area of region T is 2.25 -/
theorem area_of_region_t (r : RightRhombus) : area (region_t r) = 2.25 := by sorry

end NUMINAMATH_CALUDE_area_of_region_t_l3923_392304


namespace NUMINAMATH_CALUDE_barium_atoms_in_compound_l3923_392301

/-- The number of Barium atoms in the compound -/
def num_barium_atoms : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_oxygen_atoms : ℕ := 2

/-- The number of Hydrogen atoms in the compound -/
def num_hydrogen_atoms : ℕ := 2

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 171

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_barium : ℝ := 137.33

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_oxygen : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_hydrogen : ℝ := 1.01

theorem barium_atoms_in_compound :
  num_barium_atoms = 1 :=
sorry

end NUMINAMATH_CALUDE_barium_atoms_in_compound_l3923_392301


namespace NUMINAMATH_CALUDE_sequence_length_correct_l3923_392363

/-- The number of terms in the arithmetic sequence from 5 to 2n-1 with a common difference of 2 -/
def sequence_length (n : ℕ) : ℕ :=
  n - 2

/-- The nth term of the sequence -/
def sequence_term (n : ℕ) : ℕ :=
  2 * n + 3

theorem sequence_length_correct (n : ℕ) :
  sequence_term (sequence_length n) = 2 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_sequence_length_correct_l3923_392363


namespace NUMINAMATH_CALUDE_no_roots_implication_l3923_392357

theorem no_roots_implication (p q b c : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + p*x + q ≠ 0)
  (h2 : ∀ x : ℝ, x^2 + b*x + c ≠ 0) :
  ∀ x : ℝ, 7*x^2 + (2*p + 3*b + 4)*x + 2*q + 3*c + 2 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_roots_implication_l3923_392357


namespace NUMINAMATH_CALUDE_circle_ratio_invariant_l3923_392348

theorem circle_ratio_invariant (r : ℝ) (h : r > 2) : 
  let new_radius := r - 2
  let new_diameter := 2 * r - 4
  let new_circumference := 2 * Real.pi * new_radius
  new_circumference / new_diameter = Real.pi := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_invariant_l3923_392348


namespace NUMINAMATH_CALUDE_overlap_area_is_nine_l3923_392314

/-- Regular hexagon with area 36 -/
structure RegularHexagon :=
  (area : ℝ)
  (is_regular : Bool)
  (area_eq : area = 36)

/-- Equilateral triangle formed by connecting every other vertex of the hexagon -/
structure EquilateralTriangle (hex : RegularHexagon) :=
  (vertices : Fin 3 → Fin 6)
  (is_equilateral : Bool)
  (area : ℝ)
  (area_eq : area = hex.area / 2)

/-- The overlapping region of two equilateral triangles in the hexagon -/
def overlap_area (hex : RegularHexagon) (t1 t2 : EquilateralTriangle hex) : ℝ := sorry

/-- Theorem stating that the overlap area is 9 -/
theorem overlap_area_is_nine (hex : RegularHexagon) 
  (t1 t2 : EquilateralTriangle hex) : overlap_area hex t1 t2 = 9 := by sorry

end NUMINAMATH_CALUDE_overlap_area_is_nine_l3923_392314


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_system_l3923_392398

def inequality_system (x : ℝ) : Prop :=
  2 * x - 4 ≥ 2 ∧ 3 * x - 7 < 8

theorem solution_set_of_inequality_system :
  {x : ℝ | inequality_system x} = {x : ℝ | 3 ≤ x ∧ x < 5} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_system_l3923_392398


namespace NUMINAMATH_CALUDE_square_identities_l3923_392325

theorem square_identities (a b c : ℝ) : 
  ((a + b)^2 = a^2 + 2*a*b + b^2) ∧ 
  ((a - b)^2 = a^2 - 2*a*b + b^2) ∧ 
  ((a + b + c)^2 = a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) := by
  sorry

end NUMINAMATH_CALUDE_square_identities_l3923_392325


namespace NUMINAMATH_CALUDE_specific_group_probability_l3923_392360

-- Define the number of students in the class
def n : ℕ := 32

-- Define the number of students chosen each day
def k : ℕ := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability of selecting a specific group
def probability : ℚ := 1 / (combination n k)

-- Theorem statement
theorem specific_group_probability :
  probability = 1 / 4960 := by
  sorry

end NUMINAMATH_CALUDE_specific_group_probability_l3923_392360


namespace NUMINAMATH_CALUDE_inequality_solution_l3923_392379

theorem inequality_solution (a : ℝ) : 
  (∀ x : ℝ, (x < 1 ∨ x > 3) ↔ (a * x) / (x - 1) < 1) → 
  a = 2/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3923_392379


namespace NUMINAMATH_CALUDE_uf_championship_ratio_l3923_392361

/-- The ratio of UF's points in the championship game to their average points per game -/
theorem uf_championship_ratio : 
  ∀ (total_points : ℕ) (num_games : ℕ) (opponent_points : ℕ) (win_margin : ℕ),
    total_points = 720 →
    num_games = 24 →
    opponent_points = 11 →
    win_margin = 2 →
    (opponent_points + win_margin : ℚ) / (total_points / num_games : ℚ) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_uf_championship_ratio_l3923_392361


namespace NUMINAMATH_CALUDE_min_value_theorem_l3923_392326

/-- The circle C: (x-2)^2+(y+1)^2=5 is symmetric with respect to the line ax-by-1=0 -/
def symmetric_circle (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - 2)^2 + (y + 1)^2 = 5 ∧ a * x - b * y - 1 = 0

/-- The theorem stating the minimum value of 3/b + 2/a -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_sym : symmetric_circle a b) : 
    (∀ x y : ℝ, x > 0 → y > 0 → symmetric_circle x y → 3/y + 2/x ≥ 7 + 4 * Real.sqrt 3) ∧ 
    (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ symmetric_circle x y ∧ 3/y + 2/x = 7 + 4 * Real.sqrt 3) :=
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3923_392326


namespace NUMINAMATH_CALUDE_blood_expires_same_day_l3923_392307

/- Define the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/- Define the expiration time in seconds (8!) -/
def expiration_time : ℕ := Nat.factorial 8

/- Theorem: The blood expires in less than one day -/
theorem blood_expires_same_day : 
  (expiration_time : ℚ) / seconds_per_day < 1 := by
  sorry


end NUMINAMATH_CALUDE_blood_expires_same_day_l3923_392307


namespace NUMINAMATH_CALUDE_sine_graph_shift_l3923_392319

theorem sine_graph_shift (x : ℝ) : 
  2 * Real.sin (3 * (x - π/15) + π/5) = 2 * Real.sin (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_sine_graph_shift_l3923_392319


namespace NUMINAMATH_CALUDE_correct_assembly_rates_l3923_392341

/-- Represents the assembly and disassembly rates of coffee grinders for two robots -/
structure CoffeeGrinderRates where
  hubert_assembly : ℝ     -- Hubert's assembly rate (grinders per hour)
  robert_assembly : ℝ     -- Robert's assembly rate (grinders per hour)

/-- Checks if the given rates satisfy the problem conditions -/
def satisfies_conditions (rates : CoffeeGrinderRates) : Prop :=
  -- Each assembles four times faster than the other disassembles
  rates.hubert_assembly = 4 * (rates.robert_assembly / 4) ∧
  rates.robert_assembly = 4 * (rates.hubert_assembly / 4) ∧
  -- Morning shift conditions
  (rates.hubert_assembly - rates.robert_assembly / 4) * 3 = 27 ∧
  -- Afternoon shift conditions
  (rates.robert_assembly - rates.hubert_assembly / 4) * 6 = 120

/-- The theorem stating the correct assembly rates for Hubert and Robert -/
theorem correct_assembly_rates :
  ∃ (rates : CoffeeGrinderRates),
    satisfies_conditions rates ∧
    rates.hubert_assembly = 12 ∧
    rates.robert_assembly = 80 / 3 := by
  sorry


end NUMINAMATH_CALUDE_correct_assembly_rates_l3923_392341


namespace NUMINAMATH_CALUDE_determinant_difference_l3923_392350

theorem determinant_difference (a b c d : ℝ) :
  Matrix.det !![a, b; c, d] = 15 →
  Matrix.det !![3*a, 3*b; 3*c, 3*d] - Matrix.det !![3*b, 3*a; 3*d, 3*c] = 270 := by
sorry

end NUMINAMATH_CALUDE_determinant_difference_l3923_392350


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l3923_392336

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 2 - 3 * Complex.I) * (2 * Real.sqrt 3 + 4 * Complex.I)) = 6 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l3923_392336


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3923_392317

theorem consecutive_integers_sum (n : ℤ) : 
  (n - 1) + (n + 1) = 118 → n = 59 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3923_392317


namespace NUMINAMATH_CALUDE_x_intercept_implies_b_value_l3923_392327

/-- 
Given a line y = 2x - b with an x-intercept of 1, prove that b = 2.
-/
theorem x_intercept_implies_b_value (b : ℝ) :
  (∃ x : ℝ, x = 1 ∧ 2 * x - b = 0) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_implies_b_value_l3923_392327


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3923_392376

theorem quadratic_equation_solution (x c d : ℕ) (h1 : x^2 + 14*x = 72) 
  (h2 : x = Int.sqrt c - d) (h3 : 0 < c) (h4 : 0 < d) : c + d = 128 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3923_392376


namespace NUMINAMATH_CALUDE_sqrt_12_between_3_and_4_l3923_392375

theorem sqrt_12_between_3_and_4 : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_between_3_and_4_l3923_392375


namespace NUMINAMATH_CALUDE_marco_running_time_l3923_392369

-- Define the constants
def laps : ℕ := 7
def track_length : ℝ := 500
def first_segment : ℝ := 150
def second_segment : ℝ := 350
def speed_first : ℝ := 3
def speed_second : ℝ := 4

-- Define the theorem
theorem marco_running_time :
  let time_first := first_segment / speed_first
  let time_second := second_segment / speed_second
  let time_per_lap := time_first + time_second
  laps * time_per_lap = 962.5 := by sorry

end NUMINAMATH_CALUDE_marco_running_time_l3923_392369


namespace NUMINAMATH_CALUDE_bales_stored_l3923_392364

/-- Given the initial number of bales and the final number of bales,
    prove that Jason stored 23 bales in the barn. -/
theorem bales_stored (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 73)
  (h2 : final_bales = 96) :
  final_bales - initial_bales = 23 := by
  sorry

end NUMINAMATH_CALUDE_bales_stored_l3923_392364


namespace NUMINAMATH_CALUDE_equation_d_is_quadratic_l3923_392367

/-- A polynomial equation in x is quadratic if it can be written in the form ax² + bx + c = 0,
    where a ≠ 0 and a, b, c are constants. -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation 3(x+1)² = 2(x+1) is a quadratic equation in terms of x. -/
theorem equation_d_is_quadratic :
  is_quadratic_equation (λ x => 3 * (x + 1)^2 - 2 * (x + 1)) :=
sorry

end NUMINAMATH_CALUDE_equation_d_is_quadratic_l3923_392367


namespace NUMINAMATH_CALUDE_triangle_perimeter_max_l3923_392309

open Real

theorem triangle_perimeter_max (x : ℝ) (h : 0 < x ∧ x < 2 * π / 3) :
  let y := 4 * Real.sqrt 3 * sin (x + π / 6) + 2 * Real.sqrt 3
  ∃ (y_max : ℝ), y ≤ y_max ∧ y_max = 6 * Real.sqrt 3 := by
  sorry

#check triangle_perimeter_max

end NUMINAMATH_CALUDE_triangle_perimeter_max_l3923_392309


namespace NUMINAMATH_CALUDE_complement_of_union_l3923_392384

-- Define the universe set U
def U : Finset Nat := {0, 1, 2, 3, 4}

-- Define set M
def M : Finset Nat := {0, 4}

-- Define set N
def N : Finset Nat := {2, 4}

-- Theorem statement
theorem complement_of_union :
  (U \ (M ∪ N)) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3923_392384


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_and_custom_definition_l3923_392368

/-- The diameter of a circle with area 400π cm² is 1600 cm, given that the diameter is defined as four times the square of the radius. -/
theorem circle_diameter_from_area_and_custom_definition :
  ∀ (r d : ℝ),
  r > 0 →
  400 * Real.pi = Real.pi * r^2 →
  d = 4 * r^2 →
  d = 1600 := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_and_custom_definition_l3923_392368


namespace NUMINAMATH_CALUDE_projection_vector_a_on_b_l3923_392395

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-1, 2)

theorem projection_vector_a_on_b :
  let proj_b_a := ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b
  proj_b_a = (-3/5, 6/5) := by sorry

end NUMINAMATH_CALUDE_projection_vector_a_on_b_l3923_392395


namespace NUMINAMATH_CALUDE_average_of_combined_results_l3923_392349

theorem average_of_combined_results :
  let n₁ : ℕ := 100
  let n₂ : ℕ := 75
  let avg₁ : ℚ := 45
  let avg₂ : ℚ := 65
  let total_sum : ℚ := n₁ * avg₁ + n₂ * avg₂
  let total_count : ℕ := n₁ + n₂
  total_sum / total_count = 9375 / 175 := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l3923_392349


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3923_392351

/-- A geometric sequence with sum of first n terms Sn -/
def geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, n > 0 → a n = a 1 * r^(n-1) ∧ S n = a 1 * (1 - r^n) / (1 - r)

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  geometric_sequence a S →
  a 1 + a 3 = 5/2 →
  a 2 + a 4 = 5/4 →
  ∀ n, n > 0 → S n / a n = 2^n - 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3923_392351


namespace NUMINAMATH_CALUDE_final_result_calculation_l3923_392303

theorem final_result_calculation (chosen_number : ℤ) : 
  chosen_number = 120 → (chosen_number / 6 - 15 : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_final_result_calculation_l3923_392303


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l3923_392391

theorem similar_triangle_perimeter :
  ∀ (small_side : ℝ) (small_base : ℝ) (large_base : ℝ),
    small_side > 0 → small_base > 0 → large_base > 0 →
    small_side + small_side + small_base = 7 + 7 + 12 →
    large_base = 36 →
    large_base / small_base = 36 / 12 →
    (2 * small_side * (large_base / small_base) + large_base) = 78 :=
by
  sorry

#check similar_triangle_perimeter

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l3923_392391


namespace NUMINAMATH_CALUDE_circle_radius_proof_l3923_392387

theorem circle_radius_proof (r : ℝ) : r > 0 → 3 * (2 * π * r) = 2 * (π * r^2) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l3923_392387


namespace NUMINAMATH_CALUDE_polynomial_real_root_l3923_392318

/-- The polynomial in question -/
def polynomial (b x : ℝ) : ℝ := x^4 + b*x^3 + x^2 + b*x + 1

/-- The theorem statement -/
theorem polynomial_real_root (b : ℝ) :
  (∃ x : ℝ, polynomial b x = 0) ↔ b ≤ -1.5 := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l3923_392318


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3923_392358

theorem polynomial_simplification (q : ℝ) : 
  (4 * q^3 - 7 * q^2 + 3 * q - 2) + (5 * q^2 - 9 * q + 8) = 4 * q^3 - 2 * q^2 - 6 * q + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3923_392358


namespace NUMINAMATH_CALUDE_greatest_triangle_perimeter_l3923_392310

theorem greatest_triangle_perimeter : 
  ∀ a b c : ℕ,
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (b = 4 * a) →
  (c = 20) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (∀ x y z : ℕ, 
    (x > 0 ∧ y > 0 ∧ z > 0) →
    (y = 4 * x) →
    (z = 20) →
    (x + y > z ∧ y + z > x ∧ z + x > y) →
    (a + b + c ≥ x + y + z)) →
  a + b + c = 50 :=
by sorry

end NUMINAMATH_CALUDE_greatest_triangle_perimeter_l3923_392310


namespace NUMINAMATH_CALUDE_equation_solutions_l3923_392371

theorem equation_solutions :
  let f (x : ℝ) := (x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1)
  let g (x : ℝ) := (x - 5) * (x - 6) * (x - 5)
  ∀ x : ℝ, x ≠ 5 ∧ x ≠ 6 →
    (f x / g x = 1 ↔ x = 2 ∨ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3923_392371


namespace NUMINAMATH_CALUDE_midpoint_arrival_time_l3923_392354

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Represents a hiking event -/
structure HikingEvent where
  planned_start : Time
  planned_end : Time
  actual_start_delay : Nat
  actual_end_early : Nat

def midpoint_time (event : HikingEvent) : Time :=
  sorry

theorem midpoint_arrival_time (event : HikingEvent) : 
  event.planned_start = { hours := 10, minutes := 10 } →
  event.planned_end = { hours := 13, minutes := 10 } →
  event.actual_start_delay = 5 →
  event.actual_end_early = 4 →
  midpoint_time event = { hours := 11, minutes := 50 } :=
sorry

end NUMINAMATH_CALUDE_midpoint_arrival_time_l3923_392354


namespace NUMINAMATH_CALUDE_lemonade_parts_l3923_392394

/-- Punch mixture properties -/
structure PunchMixture where
  lemonade : ℕ
  cranberry : ℕ
  total_volume : ℕ
  ratio : ℚ

/-- Theorem: The number of parts of lemonade in the punch mixture is 12 -/
theorem lemonade_parts (p : PunchMixture) : p.lemonade = 12 :=
  by
  have h1 : p.cranberry = p.lemonade + 18 := sorry
  have h2 : p.ratio = p.lemonade / 5 := sorry
  have h3 : p.total_volume = 72 := sorry
  have h4 : p.lemonade + p.cranberry = p.total_volume := sorry
  sorry

#check lemonade_parts

end NUMINAMATH_CALUDE_lemonade_parts_l3923_392394


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_plus_minus_one_l3923_392306

theorem fraction_zero_implies_x_plus_minus_one (x : ℝ) :
  (x^2 - 1) / x = 0 → x ≠ 0 → (x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_plus_minus_one_l3923_392306


namespace NUMINAMATH_CALUDE_left_handed_fraction_l3923_392362

theorem left_handed_fraction (red blue : ℕ) (h_ratio : red = blue) 
  (h_red_left : red / 3 = red.div 3) 
  (h_blue_left : 2 * (blue / 3) = blue.div 3 * 2) : 
  (red.div 3 + blue.div 3 * 2) / (red + blue) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_fraction_l3923_392362


namespace NUMINAMATH_CALUDE_workshop_attendance_l3923_392356

theorem workshop_attendance : 
  ∀ (total wolf_laureates wolf_and_nobel_laureates nobel_laureates : ℕ),
    wolf_laureates = 31 →
    wolf_and_nobel_laureates = 16 →
    nobel_laureates = 27 →
    ∃ (non_wolf_nobel non_wolf_non_nobel : ℕ),
      non_wolf_nobel = nobel_laureates - wolf_and_nobel_laureates ∧
      non_wolf_nobel = non_wolf_non_nobel + 3 ∧
      total = wolf_laureates + non_wolf_nobel + non_wolf_non_nobel →
      total = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_attendance_l3923_392356


namespace NUMINAMATH_CALUDE_triangle_area_l3923_392312

theorem triangle_area (a b c : ℝ) (ha : a^2 = 225) (hb : b^2 = 225) (hc : c^2 = 64) :
  (1/2) * a * c = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3923_392312


namespace NUMINAMATH_CALUDE_integer_tuple_solution_l3923_392382

theorem integer_tuple_solution :
  ∀ a b c x y z : ℕ,
    a + b + c = x * y * z →
    x + y + z = a * b * c →
    a ≥ b →
    b ≥ c →
    c ≥ 1 →
    x ≥ y →
    y ≥ z →
    z ≥ 1 →
    ((a = 2 ∧ b = 2 ∧ c = 2 ∧ x = 6 ∧ y = 1 ∧ z = 1) ∨
     (a = 5 ∧ b = 2 ∧ c = 1 ∧ x = 8 ∧ y = 1 ∧ z = 1) ∨
     (a = 3 ∧ b = 3 ∧ c = 1 ∧ x = 7 ∧ y = 1 ∧ z = 1) ∨
     (a = 3 ∧ b = 2 ∧ c = 1 ∧ x = 3 ∧ y = 2 ∧ z = 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_integer_tuple_solution_l3923_392382


namespace NUMINAMATH_CALUDE_problem_solution_l3923_392355

-- Define the sets A and B
def A (a b c : ℝ) : Prop := a^2 - b*c - 8*a + 7 = 0
def B (a b c : ℝ) : Prop := b^2 + c^2 + b*c - b*a + b = 0

-- Define the function y
def y (a b c : ℝ) : ℝ := a*b + b*c + a*c

-- Theorem statement
theorem problem_solution :
  ∃ (a b c : ℝ), A a b c ∧ B a b c →
  (∀ a : ℝ, (∃ b c : ℝ, A a b c ∧ B a b c) → 1 ≤ a ∧ a ≤ 9) ∧
  (∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, 
    A a₁ b₁ c₁ ∧ B a₁ b₁ c₁ ∧ A a₂ b₂ c₂ ∧ B a₂ b₂ c₂ ∧
    y a₁ b₁ c₁ = 88 ∧ y a₂ b₂ c₂ = -56 ∧
    ∀ a b c : ℝ, A a b c ∧ B a b c → -56 ≤ y a b c ∧ y a b c ≤ 88) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3923_392355


namespace NUMINAMATH_CALUDE_gcd_378_90_l3923_392315

theorem gcd_378_90 : Nat.gcd 378 90 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_378_90_l3923_392315


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l3923_392329

theorem slower_speed_calculation (actual_distance : ℝ) (faster_speed : ℝ) (additional_distance : ℝ)
  (h1 : actual_distance = 50)
  (h2 : faster_speed = 14)
  (h3 : additional_distance = 20) :
  ∃ x : ℝ, x > 0 ∧ actual_distance / x = (actual_distance + additional_distance) / faster_speed ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l3923_392329


namespace NUMINAMATH_CALUDE_perimeter_plus_area_equals_9_sqrt_41_l3923_392302

/-- A parallelogram with integer coordinates -/
structure Parallelogram where
  a : ℤ × ℤ
  b : ℤ × ℤ
  c : ℤ × ℤ
  d : ℤ × ℤ

/-- The specific parallelogram from the problem -/
def specificParallelogram : Parallelogram :=
  { a := (0, 0),
    b := (4, 5),
    c := (11, 5),
    d := (7, 0) }

/-- Calculate the perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ :=
  sorry

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem perimeter_plus_area_equals_9_sqrt_41 :
  perimeter specificParallelogram + area specificParallelogram = 9 * Real.sqrt 41 :=
sorry

end NUMINAMATH_CALUDE_perimeter_plus_area_equals_9_sqrt_41_l3923_392302


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3923_392311

theorem absolute_value_inequality (x : ℝ) : 
  abs (x - 1) + abs (x + 2) < 5 ↔ -3 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3923_392311


namespace NUMINAMATH_CALUDE_prime_factors_count_l3923_392389

/-- The number of positive divisors of n -/
def d (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The main expression in the problem -/
def f (n : ℕ) : ℕ := (n^(2*n) + n^n + n + 1)^(2*n) + (n^(2*n) + n^n + n + 1)^n + 1

/-- The theorem statement -/
theorem prime_factors_count (n : ℕ) (h : ¬3 ∣ n) : 
  2 * d n ≤ (Nat.factors (f n)).card := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_count_l3923_392389


namespace NUMINAMATH_CALUDE_cost_price_of_toy_cost_price_is_1300_l3923_392393

/-- The cost price of a toy given the selling conditions -/
theorem cost_price_of_toy (num_toys : ℕ) (selling_price : ℕ) (gain_toys : ℕ) : ℕ :=
  let cost_price := selling_price / (num_toys + gain_toys)
  cost_price

/-- Proof that the cost price of a toy is 1300 under given conditions -/
theorem cost_price_is_1300 : cost_price_of_toy 18 27300 3 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_toy_cost_price_is_1300_l3923_392393


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l3923_392370

theorem smallest_square_containing_circle (r : ℝ) (h : r = 5) :
  (2 * r) ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l3923_392370


namespace NUMINAMATH_CALUDE_second_train_length_calculation_l3923_392335

/-- Calculates the length of the second train given the speeds of two trains,
    the length of the first train, and the time they take to clear each other. -/
def second_train_length (speed1 speed2 : ℝ) (length1 time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let distance := relative_speed * time
  distance - length1

theorem second_train_length_calculation :
  let speed1 := 42 * (1000 / 3600)  -- Convert 42 kmph to m/s
  let speed2 := 30 * (1000 / 3600)  -- Convert 30 kmph to m/s
  let length1 := 200
  let time := 23.998
  abs (second_train_length speed1 speed2 length1 time - 279.96) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_second_train_length_calculation_l3923_392335


namespace NUMINAMATH_CALUDE_polynomial_value_l3923_392305

theorem polynomial_value (x y : ℝ) (h : 2 * x^2 + 3 * y + 7 = 8) :
  -2 * x^2 - 3 * y + 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l3923_392305


namespace NUMINAMATH_CALUDE_base_r_transaction_l3923_392396

/-- Represents a number in base r --/
def BaseR (digits : List Nat) (r : Nat) : Nat :=
  digits.foldr (fun d acc => d + r * acc) 0

/-- The problem statement --/
theorem base_r_transaction (r : Nat) : 
  (BaseR [5, 2, 1] r) + (BaseR [1, 1, 0] r) - (BaseR [3, 7, 1] r) = (BaseR [1, 0, 0, 2] r) →
  r^3 - 3*r^2 + 4*r - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_r_transaction_l3923_392396


namespace NUMINAMATH_CALUDE_ellipse_m_value_l3923_392308

/-- An ellipse with equation x²/10 + y²/m = 1, foci on y-axis, and major axis length 8 has m = 16 -/
theorem ellipse_m_value (m : ℝ) : 
  (∀ x y : ℝ, x^2 / 10 + y^2 / m = 1) →  -- Ellipse equation
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 = 10 ∧ b^2 = m) →  -- Standard form of ellipse
  (∀ x : ℝ, x^2 / 10 + 0^2 / m ≠ 1) →  -- Foci on y-axis
  (2 * Real.sqrt m = 8) →  -- Major axis length
  m = 16 := by
sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l3923_392308


namespace NUMINAMATH_CALUDE_percentage_watching_two_shows_l3923_392334

def total_residents : ℕ := 600
def watch_island_survival : ℕ := (35 * total_residents) / 100
def watch_lovelost_lawyers : ℕ := (40 * total_residents) / 100
def watch_medical_emergency : ℕ := (50 * total_residents) / 100
def watch_all_three : ℕ := 21

theorem percentage_watching_two_shows :
  let watch_two_shows := watch_island_survival + watch_lovelost_lawyers + watch_medical_emergency - total_residents + watch_all_three
  (watch_two_shows : ℚ) / total_residents * 100 = 285 / 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_watching_two_shows_l3923_392334


namespace NUMINAMATH_CALUDE_cos_120_degrees_l3923_392386

theorem cos_120_degrees : Real.cos (2 * π / 3) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l3923_392386


namespace NUMINAMATH_CALUDE_power_product_rule_l3923_392392

theorem power_product_rule (a : ℝ) : a^3 * a^5 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_power_product_rule_l3923_392392


namespace NUMINAMATH_CALUDE_square_sum_bound_l3923_392374

theorem square_sum_bound (a b c : ℝ) :
  (|a^2 + b + c| + |a + b^2 - c| ≤ 1) → (a^2 + b^2 + c^2 < 100) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_bound_l3923_392374


namespace NUMINAMATH_CALUDE_problem_statement_l3923_392332

theorem problem_statement (a b c p q : ℕ) (hp : p > q) 
  (h_sum : a + b + c = 2 * p * q * (p^30 + q^30)) : 
  let k := a^3 + b^3 + c^3
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ k = x * y ∧ 
  (∀ (a' b' c' : ℕ), a' + b' + c' = 2 * p * q * (p^30 + q^30) → 
    a' * b' * c' ≤ a * b * c → 1984 ∣ k) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3923_392332


namespace NUMINAMATH_CALUDE_benny_attended_games_l3923_392390

/-- 
Given:
- The total number of baseball games is 39.
- Benny missed 25 games.

Prove that the number of games Benny attended is 14.
-/
theorem benny_attended_games (total_games : ℕ) (missed_games : ℕ) 
  (h1 : total_games = 39)
  (h2 : missed_games = 25) :
  total_games - missed_games = 14 := by
  sorry

end NUMINAMATH_CALUDE_benny_attended_games_l3923_392390


namespace NUMINAMATH_CALUDE_cos_135_degrees_l3923_392399

theorem cos_135_degrees : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l3923_392399


namespace NUMINAMATH_CALUDE_object_distances_l3923_392340

-- Define the parameters
def speed1 : ℝ := 3
def speed2 : ℝ := 4
def initial_distance : ℝ := 20
def final_distance : ℝ := 10
def time_elapsed : ℝ := 2

-- Define the theorem
theorem object_distances (x y : ℝ) :
  -- Conditions
  (x^2 + y^2 = initial_distance^2) →
  ((x - speed1 * time_elapsed)^2 + (y - speed2 * time_elapsed)^2 = final_distance^2) →
  -- Conclusion
  (x = 12 ∧ y = 16) :=
by sorry

end NUMINAMATH_CALUDE_object_distances_l3923_392340


namespace NUMINAMATH_CALUDE_loading_dock_problem_l3923_392323

/-- Proves that given the conditions of the loading dock problem, 
    the fraction of boxes loaded by each night crew worker 
    compared to each day crew worker is 5/14 -/
theorem loading_dock_problem 
  (day_crew : ℕ) 
  (night_crew : ℕ) 
  (h1 : night_crew = (4 : ℚ) / 5 * day_crew) 
  (h2 : (5 : ℚ) / 7 = day_crew_boxes / total_boxes) 
  (day_crew_boxes : ℚ) 
  (night_crew_boxes : ℚ) 
  (total_boxes : ℚ) 
  (h3 : total_boxes = day_crew_boxes + night_crew_boxes) 
  (h4 : total_boxes ≠ 0) 
  (h5 : day_crew ≠ 0) 
  (h6 : night_crew ≠ 0) :
  (night_crew_boxes / night_crew) / (day_crew_boxes / day_crew) = (5 : ℚ) / 14 := by
  sorry

end NUMINAMATH_CALUDE_loading_dock_problem_l3923_392323


namespace NUMINAMATH_CALUDE_range_of_m_value_of_m_l3923_392383

-- Define the quadratic equation
def quadratic_eq (m x : ℝ) : Prop :=
  x^2 - 2*(1-m)*x + m^2 = 0

-- Define the roots of the equation
def roots (m x₁ x₂ : ℝ) : Prop :=
  quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ x₁ ≠ x₂

-- Define the additional condition
def additional_condition (m x₁ x₂ : ℝ) : Prop :=
  x₁^2 + 12*m + x₂^2 = 10

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∃ x₁ x₂, roots m x₁ x₂) → m ≤ 1/2 :=
sorry

-- Theorem for the value of m given the additional condition
theorem value_of_m (m x₁ x₂ : ℝ) :
  roots m x₁ x₂ → additional_condition m x₁ x₂ → m = -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_value_of_m_l3923_392383


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3923_392388

theorem trigonometric_identity (α : ℝ) :
  Real.sin (4 * α) - Real.sin (5 * α) - Real.sin (6 * α) + Real.sin (7 * α) =
  -4 * Real.sin (α / 2) * Real.sin α * Real.sin (11 * α / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3923_392388


namespace NUMINAMATH_CALUDE_exam_average_problem_l3923_392378

theorem exam_average_problem :
  ∀ (N : ℕ),
  (15 : ℝ) * 70 + (10 : ℝ) * 95 = (N : ℝ) * 80 →
  N = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_average_problem_l3923_392378


namespace NUMINAMATH_CALUDE_shopping_tax_calculation_l3923_392344

-- Define the percentages of spending
def clothing_percent : ℝ := 0.50
def food_percent : ℝ := 0.20
def other_percent : ℝ := 0.30

-- Define the tax rates
def clothing_tax_rate : ℝ := 0.04
def food_tax_rate : ℝ := 0
def total_tax_rate : ℝ := 0.044

-- Define the unknown tax rate on other items
def other_tax_rate : ℝ := sorry

theorem shopping_tax_calculation :
  let total_spent := 100  -- Assume total spent is 100 for simplicity
  let clothing_tax := clothing_percent * total_spent * clothing_tax_rate
  let other_tax := other_percent * total_spent * other_tax_rate
  clothing_tax + other_tax = total_tax_rate * total_spent →
  other_tax_rate = 0.08 := by sorry

end NUMINAMATH_CALUDE_shopping_tax_calculation_l3923_392344


namespace NUMINAMATH_CALUDE_slope_condition_l3923_392365

-- Define the function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x

-- Define the theorem
theorem slope_condition (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : f m x₁ = y₁)
  (h2 : f m x₂ = y₂)
  (h3 : x₁ > x₂)
  (h4 : y₁ > y₂) :
  m > 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_condition_l3923_392365


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l3923_392380

theorem set_equality_implies_sum (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2013 + b^2013 = -1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l3923_392380


namespace NUMINAMATH_CALUDE_chewing_gum_revenue_comparison_l3923_392366

theorem chewing_gum_revenue_comparison 
  (last_year_revenue : ℝ) 
  (projected_increase_rate : ℝ) 
  (actual_decrease_rate : ℝ) 
  (h1 : projected_increase_rate = 0.25)
  (h2 : actual_decrease_rate = 0.25) :
  (last_year_revenue * (1 - actual_decrease_rate)) / 
  (last_year_revenue * (1 + projected_increase_rate)) = 0.6 := by
sorry

end NUMINAMATH_CALUDE_chewing_gum_revenue_comparison_l3923_392366


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l3923_392316

theorem sin_thirteen_pi_sixths : Real.sin (13 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l3923_392316


namespace NUMINAMATH_CALUDE_blue_shoes_count_l3923_392373

theorem blue_shoes_count (total : ℕ) (purple : ℕ) (h1 : total = 1250) (h2 : purple = 355) :
  ∃ (blue green : ℕ), blue + green + purple = total ∧ green = purple ∧ blue = 540 := by
  sorry

end NUMINAMATH_CALUDE_blue_shoes_count_l3923_392373


namespace NUMINAMATH_CALUDE_curve_ellipse_equivalence_l3923_392338

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y+4)^2) + Real.sqrt (x^2 + (y-4)^2) = 10

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  y^2/25 + x^2/9 = 1

-- Theorem stating the equivalence of the two equations
theorem curve_ellipse_equivalence :
  ∀ x y : ℝ, curve_equation x y ↔ ellipse_equation x y :=
by sorry

end NUMINAMATH_CALUDE_curve_ellipse_equivalence_l3923_392338


namespace NUMINAMATH_CALUDE_interest_rate_is_four_percent_l3923_392372

/-- Proves that the rate of interest is 4% per annum given the conditions of the problem -/
theorem interest_rate_is_four_percent (principal : ℝ) (simple_interest : ℝ) (time : ℝ) 
  (h1 : simple_interest = principal - 2080)
  (h2 : principal = 2600)
  (h3 : time = 5)
  (h4 : simple_interest = (principal * rate * time) / 100) : rate = 4 := by
  sorry

#check interest_rate_is_four_percent

end NUMINAMATH_CALUDE_interest_rate_is_four_percent_l3923_392372


namespace NUMINAMATH_CALUDE_student_grade_average_l3923_392345

theorem student_grade_average (grade1 grade2 : ℝ) 
  (h1 : grade1 = 70)
  (h2 : grade2 = 80) : 
  ∃ (grade3 : ℝ), (grade1 + grade2 + grade3) / 3 = grade3 ∧ grade3 = 75 := by
sorry

end NUMINAMATH_CALUDE_student_grade_average_l3923_392345


namespace NUMINAMATH_CALUDE_total_rooms_to_paint_l3923_392339

theorem total_rooms_to_paint 
  (time_per_room : ℕ) 
  (rooms_painted : ℕ) 
  (time_remaining : ℕ) : 
  time_per_room = 7 → 
  rooms_painted = 2 → 
  time_remaining = 63 → 
  rooms_painted + (time_remaining / time_per_room) = 11 :=
by sorry

end NUMINAMATH_CALUDE_total_rooms_to_paint_l3923_392339


namespace NUMINAMATH_CALUDE_xyz_inequality_l3923_392346

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2*x*y*z = 1) :
  8*x*y*z ≤ 1 ∧
  (8*x*y*z = 1 ↔
    (x, y, z) = (1/2, 1/2, 1/2) ∨
    (x, y, z) = (-1/2, -1/2, 1/2) ∨
    (x, y, z) = (-1/2, 1/2, -1/2) ∨
    (x, y, z) = (1/2, -1/2, -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_xyz_inequality_l3923_392346


namespace NUMINAMATH_CALUDE_not_necessary_nor_sufficient_condition_l3923_392352

theorem not_necessary_nor_sufficient_condition (m n : ℕ+) :
  ¬(∀ a b : ℝ, (a^m.val - b^m.val) * (a^n.val - b^n.val) > 0 → a > b) ∧
  ¬(∀ a b : ℝ, a > b → (a^m.val - b^m.val) * (a^n.val - b^n.val) > 0) :=
by sorry

end NUMINAMATH_CALUDE_not_necessary_nor_sufficient_condition_l3923_392352


namespace NUMINAMATH_CALUDE_farm_cows_count_l3923_392324

/-- The total number of cows on the farm -/
def total_cows : ℕ := 140

/-- The percentage of cows with a red spot -/
def red_spot_percentage : ℚ := 40 / 100

/-- The percentage of cows without a red spot that have a blue spot -/
def blue_spot_percentage : ℚ := 25 / 100

/-- The number of cows with no spot -/
def no_spot_cows : ℕ := 63

theorem farm_cows_count :
  (total_cows : ℚ) * (1 - red_spot_percentage) * (1 - blue_spot_percentage) = no_spot_cows :=
sorry

end NUMINAMATH_CALUDE_farm_cows_count_l3923_392324


namespace NUMINAMATH_CALUDE_six_digit_integers_count_l3923_392313

/-- The number of different positive, six-digit integers that can be formed
    using the digits 2, 2, 2, 5, 5, and 9 -/
def six_digit_integers : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)

/-- Theorem stating that the number of different positive, six-digit integers
    that can be formed using the digits 2, 2, 2, 5, 5, and 9 is equal to 60 -/
theorem six_digit_integers_count : six_digit_integers = 60 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_integers_count_l3923_392313


namespace NUMINAMATH_CALUDE_quinn_reading_rate_l3923_392321

/-- A reading challenge that lasts for a certain number of weeks -/
structure ReadingChallenge where
  duration : ℕ  -- Duration of the challenge in weeks
  books_per_coupon : ℕ  -- Number of books required for one coupon

/-- A participant in the reading challenge -/
structure Participant where
  challenge : ReadingChallenge
  coupons_earned : ℕ  -- Number of coupons earned

def books_per_week (p : Participant) : ℚ :=
  (p.coupons_earned * p.challenge.books_per_coupon : ℚ) / p.challenge.duration

theorem quinn_reading_rate (c : ReadingChallenge) (p : Participant) :
    c.duration = 10 ∧ c.books_per_coupon = 5 ∧ p.challenge = c ∧ p.coupons_earned = 4 →
    books_per_week p = 2 := by
  sorry

end NUMINAMATH_CALUDE_quinn_reading_rate_l3923_392321


namespace NUMINAMATH_CALUDE_triangle_property_l3923_392342

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states that for a triangle satisfying certain conditions, 
    angle B is 2π/3 and the area is (3√3)/2. -/
theorem triangle_property (t : Triangle) 
  (h1 : t.a * Real.sin t.B + t.b * Real.sin t.A = t.b * Real.sin t.C - t.c * Real.sin t.B)
  (h2 : t.b = Real.sqrt 13)
  (h3 : t.a + t.c = 4) : 
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l3923_392342


namespace NUMINAMATH_CALUDE_megatek_graph_is_pie_chart_l3923_392300

-- Define the properties of the graph
structure EmployeeGraph where
  -- The graph is circular
  isCircular : Bool
  -- The angle of each sector is proportional to the quantity it represents
  isSectorProportional : Bool
  -- The manufacturing sector angle
  manufacturingAngle : ℝ
  -- The percentage of employees in manufacturing
  manufacturingPercentage : ℝ

-- Define a pie chart
def isPieChart (graph : EmployeeGraph) : Prop :=
  graph.isCircular ∧ 
  graph.isSectorProportional ∧
  graph.manufacturingAngle = 144 ∧
  graph.manufacturingPercentage = 40

-- Theorem to prove
theorem megatek_graph_is_pie_chart (graph : EmployeeGraph) 
  (h1 : graph.isCircular = true)
  (h2 : graph.isSectorProportional = true)
  (h3 : graph.manufacturingAngle = 144)
  (h4 : graph.manufacturingPercentage = 40) :
  isPieChart graph :=
sorry

end NUMINAMATH_CALUDE_megatek_graph_is_pie_chart_l3923_392300


namespace NUMINAMATH_CALUDE_turkey_cost_per_employee_l3923_392381

/-- The cost of turkeys for employees --/
theorem turkey_cost_per_employee (num_employees : ℕ) (total_cost : ℚ) : 
  num_employees = 85 → total_cost = 2125 → (total_cost / num_employees : ℚ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_turkey_cost_per_employee_l3923_392381


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l3923_392343

/-- A line represented by y = mx + b -/
structure Line where
  m : ℚ
  b : ℚ

/-- The point of intersection of two lines -/
def intersection (l1 l2 : Line) : ℚ × ℚ :=
  let x := (l2.b - l1.b) / (l1.m - l2.m)
  let y := l1.m * x + l1.b
  (x, y)

/-- Theorem: If three lines intersect at a single point, and two of them are
    y = 3x + 5 and y = -5x + 20, then the third line y = 4x + p must have p = 25/8 -/
theorem intersection_of_three_lines
  (l1 : Line)
  (l2 : Line)
  (l3 : Line)
  (h1 : l1 = ⟨3, 5⟩)
  (h2 : l2 = ⟨-5, 20⟩)
  (h3 : l3.m = 4)
  (h_intersect : intersection l1 l2 = intersection l2 l3) :
  l3.b = 25/8 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l3923_392343


namespace NUMINAMATH_CALUDE_sally_picked_42_peaches_l3923_392385

/-- The number of peaches Sally picked -/
def peaches_picked (initial peaches_now : ℕ) : ℕ := peaches_now - initial

/-- Theorem stating that Sally picked 42 peaches -/
theorem sally_picked_42_peaches : peaches_picked 13 55 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sally_picked_42_peaches_l3923_392385


namespace NUMINAMATH_CALUDE_max_expression_value_l3923_392320

def expression (a b c d : ℕ) : ℕ := d * (c^a - b)

theorem max_expression_value :
  ∃ (a b c d : ℕ),
    a ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    b ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    c ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    d ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    expression a b c d = 126 ∧
    ∀ (x y z w : ℕ),
      x ∈ ({1, 2, 3, 4} : Set ℕ) →
      y ∈ ({1, 2, 3, 4} : Set ℕ) →
      z ∈ ({1, 2, 3, 4} : Set ℕ) →
      w ∈ ({1, 2, 3, 4} : Set ℕ) →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
      expression x y z w ≤ 126 :=
by
  sorry


end NUMINAMATH_CALUDE_max_expression_value_l3923_392320


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_theorem_l3923_392337

def ellipse_eccentricity_range (a b c : ℝ) (F₁ F₂ P : ℝ × ℝ) : Prop :=
  let e := c / a
  0 < b ∧ b < a ∧
  c^2 + b^2 = a^2 ∧
  F₁ = (-c, 0) ∧
  F₂ = (c, 0) ∧
  P.1 = a^2 / c ∧
  (∃ m : ℝ, P = (a^2 / c, m) ∧
    let K := ((a^2 - c^2) / (2 * c), m / 2)
    (P.2 - F₁.2) * (K.2 - F₂.2) = -(P.1 - F₁.1) * (K.1 - F₂.1)) →
  Real.sqrt 3 / 3 ≤ e ∧ e < 1

theorem ellipse_eccentricity_theorem (a b c : ℝ) (F₁ F₂ P : ℝ × ℝ) :
  ellipse_eccentricity_range a b c F₁ F₂ P := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_theorem_l3923_392337


namespace NUMINAMATH_CALUDE_janeles_cats_average_weight_is_correct_l3923_392347

/-- The combined average weight of Janele's cats -/
def janeles_cats_average_weight : ℝ := by sorry

/-- The weights of Janele's first 7 cats -/
def first_seven_cats_weights : List ℝ := [12, 12, 14.7, 9.3, 14.9, 15.6, 8.7]

/-- Lily's weights over 4 days -/
def lily_weights : List ℝ := [14, 15.3, 13.2, 14.7]

/-- The number of Janele's cats -/
def num_cats : ℕ := 8

theorem janeles_cats_average_weight_is_correct :
  janeles_cats_average_weight = 
    (List.sum first_seven_cats_weights + List.sum lily_weights / 4) / num_cats := by sorry

end NUMINAMATH_CALUDE_janeles_cats_average_weight_is_correct_l3923_392347


namespace NUMINAMATH_CALUDE_rectangle_area_l3923_392333

theorem rectangle_area (x : ℝ) (h : x > 0) :
  ∃ (w : ℝ), w > 0 ∧ 
  w^2 + (3*w)^2 = x^2 ∧ 
  3*w^2 = (3/10)*x^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3923_392333


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l3923_392328

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_tenth_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_third : a 3 = 23)
  (h_seventh : a 7 = 35) :
  a 10 = 44 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l3923_392328


namespace NUMINAMATH_CALUDE_carrie_pays_94_l3923_392359

/-- The amount Carrie pays for clothes given the quantities and prices of items, and that her mom pays half the total cost. -/
def carriePays (shirtQuantity pantQuantity jacketQuantity : ℕ) 
               (shirtPrice pantPrice jacketPrice : ℚ) : ℚ :=
  let totalCost := shirtQuantity * shirtPrice + 
                   pantQuantity * pantPrice + 
                   jacketQuantity * jacketPrice
  totalCost / 2

/-- Theorem stating that Carrie pays $94 for the clothes. -/
theorem carrie_pays_94 : 
  carriePays 4 2 2 8 18 60 = 94 := by
  sorry

end NUMINAMATH_CALUDE_carrie_pays_94_l3923_392359
