import Mathlib

namespace NUMINAMATH_CALUDE_solve_equation_l630_63080

-- Define the custom operation *
def star (a b : ℚ) : ℚ := 4 * a - 2 * b

-- State the theorem
theorem solve_equation : ∃ x : ℚ, star 3 (star 6 x) = 2 ∧ x = 19/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l630_63080


namespace NUMINAMATH_CALUDE_x_value_l630_63078

theorem x_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l630_63078


namespace NUMINAMATH_CALUDE_problem_solution_l630_63026

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.log x / Real.log y + Real.log y / Real.log x = 4)
  (h2 : x * y = 64) :
  (x + y) / 2 = 13 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l630_63026


namespace NUMINAMATH_CALUDE_max_score_is_six_l630_63057

/-- Represents a 5x5 game board -/
def GameBoard : Type := Fin 5 → Fin 5 → Bool

/-- Calculates the sum of a 3x3 sub-square starting at (i, j) -/
def subSquareSum (board : GameBoard) (i j : Fin 3) : ℕ :=
  (Finset.sum (Finset.range 3) fun x =>
    Finset.sum (Finset.range 3) fun y =>
      if board (i + x) (j + y) then 1 else 0)

/-- Calculates the score of a given board (maximum sum of any 3x3 sub-square) -/
def boardScore (board : GameBoard) : ℕ :=
  Finset.sup (Finset.range 3) fun i =>
    Finset.sup (Finset.range 3) fun j =>
      subSquareSum board i j

/-- Represents a strategy for Player 2 -/
def Player2Strategy : Type := GameBoard → Fin 5 → Fin 5

/-- Represents the game play with both players' moves -/
def gamePlay (p2strat : Player2Strategy) : GameBoard :=
  sorry -- Implementation of game play

theorem max_score_is_six :
  ∀ (p2strat : Player2Strategy),
    boardScore (gamePlay p2strat) ≤ 6 ∧
    ∃ (optimal_p2strat : Player2Strategy),
      boardScore (gamePlay optimal_p2strat) = 6 :=
sorry

end NUMINAMATH_CALUDE_max_score_is_six_l630_63057


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l630_63025

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → 
    (a : ℕ) + (b : ℕ) ≥ (x : ℕ) + (y : ℕ)) → 
  (x : ℕ) + (y : ℕ) = 64 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l630_63025


namespace NUMINAMATH_CALUDE_power_sum_integer_l630_63093

theorem power_sum_integer (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/(x^n) = m :=
sorry

end NUMINAMATH_CALUDE_power_sum_integer_l630_63093


namespace NUMINAMATH_CALUDE_triangle_angle_from_cosine_relation_l630_63029

theorem triangle_angle_from_cosine_relation (a b c : ℝ) (A B C : ℝ) :
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (2 * b * Real.cos B = a * Real.cos C + c * Real.cos A) →
  B = π / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_from_cosine_relation_l630_63029


namespace NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l630_63023

/-- The volume of a tetrahedron formed by non-adjacent vertices of a cube -/
theorem tetrahedron_volume_in_cube (cube_side : ℝ) (h : cube_side = 8) :
  let tetrahedron_volume := (cube_side^3 * Real.sqrt 2) / 3
  tetrahedron_volume = (512 * Real.sqrt 2) / 3 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l630_63023


namespace NUMINAMATH_CALUDE_square_equals_product_sum_solutions_l630_63002

theorem square_equals_product_sum_solutions :
  ∀ (a b : ℤ), a ≥ 0 → b ≥ 0 → a^2 = b * (b + 7) → (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end NUMINAMATH_CALUDE_square_equals_product_sum_solutions_l630_63002


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l630_63062

theorem min_value_of_fraction (a : ℝ) (h : a > 1) :
  (a^2 - a + 1) / (a - 1) ≥ 3 ∧
  ∃ b > 1, (b^2 - b + 1) / (b - 1) = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l630_63062


namespace NUMINAMATH_CALUDE_mariels_dogs_count_l630_63054

/-- The number of dogs Mariel is walking -/
def mariels_dogs : ℕ :=
  let total_legs : ℕ := 36
  let num_walkers : ℕ := 2
  let other_walker_dogs : ℕ := 3
  let human_legs : ℕ := 2
  let dog_legs : ℕ := 4
  let total_dogs : ℕ := (total_legs - num_walkers * human_legs) / dog_legs
  total_dogs - other_walker_dogs

theorem mariels_dogs_count : mariels_dogs = 5 := by
  sorry

end NUMINAMATH_CALUDE_mariels_dogs_count_l630_63054


namespace NUMINAMATH_CALUDE_range_of_product_l630_63066

theorem range_of_product (x y z : ℝ) 
  (hx : -3 < x) (hxy : x < y) (hy : y < 1) 
  (hz1 : -4 < z) (hz2 : z < 0) : 
  0 < (x - y) * z ∧ (x - y) * z < 16 := by
  sorry

end NUMINAMATH_CALUDE_range_of_product_l630_63066


namespace NUMINAMATH_CALUDE_royal_family_theorem_l630_63076

/-- Represents the royal family -/
structure RoyalFamily where
  king_age : ℕ
  queen_age : ℕ
  num_sons : ℕ
  num_daughters : ℕ
  children_total_age : ℕ

/-- The conditions of the problem -/
def royal_family_conditions (family : RoyalFamily) : Prop :=
  family.king_age = 35 ∧
  family.queen_age = 35 ∧
  family.num_sons = 3 ∧
  family.num_daughters ≥ 1 ∧
  family.children_total_age = 35 ∧
  family.num_sons + family.num_daughters ≤ 20

/-- The theorem to be proved -/
theorem royal_family_theorem (family : RoyalFamily) 
  (h : royal_family_conditions family) :
  family.num_sons + family.num_daughters = 7 ∨
  family.num_sons + family.num_daughters = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_royal_family_theorem_l630_63076


namespace NUMINAMATH_CALUDE_sum_reciprocals_squared_l630_63060

-- Define the constants
noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6

-- State the theorem
theorem sum_reciprocals_squared :
  (1/a + 1/b + 1/c + 1/d)^2 = 96/529 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_squared_l630_63060


namespace NUMINAMATH_CALUDE_caterer_sundae_order_l630_63044

/-- Represents the problem of determining the number of sundaes ordered by a caterer --/
theorem caterer_sundae_order (total_price : ℚ) (ice_cream_bars : ℕ) (ice_cream_price : ℚ) (sundae_price : ℚ)
  (h1 : total_price = 200)
  (h2 : ice_cream_bars = 225)
  (h3 : ice_cream_price = 60/100)
  (h4 : sundae_price = 52/100) :
  ∃ (sundaes : ℕ), sundaes = 125 ∧ total_price = ice_cream_bars * ice_cream_price + sundaes * sundae_price :=
by sorry

end NUMINAMATH_CALUDE_caterer_sundae_order_l630_63044


namespace NUMINAMATH_CALUDE_min_sum_abs_l630_63063

theorem min_sum_abs (x : ℝ) : 
  |x + 1| + |x + 3| + |x + 6| ≥ 5 ∧ ∃ y : ℝ, |y + 1| + |y + 3| + |y + 6| = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_abs_l630_63063


namespace NUMINAMATH_CALUDE_ferry_tourists_l630_63005

theorem ferry_tourists (a₁ : ℕ) (d : ℕ) (n : ℕ) (h1 : a₁ = 85) (h2 : d = 3) (h3 : n = 5) :
  n * (2 * a₁ + (n - 1) * d) / 2 = 455 := by
  sorry

end NUMINAMATH_CALUDE_ferry_tourists_l630_63005


namespace NUMINAMATH_CALUDE_kens_climbing_pace_l630_63094

/-- Climbing problem -/
theorem kens_climbing_pace 
  (sari_head_start : ℝ)  -- Time Sari starts before Ken (in hours)
  (sari_initial_lead : ℝ)  -- Sari's lead when Ken starts (in meters)
  (ken_climbing_time : ℝ)  -- Time Ken spends climbing (in hours)
  (final_distance : ℝ)  -- Distance Ken is ahead of Sari at the summit (in meters)
  (h1 : sari_head_start = 2)
  (h2 : sari_initial_lead = 700)
  (h3 : ken_climbing_time = 5)
  (h4 : final_distance = 50) :
  (sari_initial_lead + final_distance) / ken_climbing_time + 
  (sari_initial_lead / sari_head_start) = 500 :=
sorry

end NUMINAMATH_CALUDE_kens_climbing_pace_l630_63094


namespace NUMINAMATH_CALUDE_identity_function_characterization_l630_63089

theorem identity_function_characterization (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y)
  (h_one : f 1 = 1)
  (h_additive : ∀ x y, f (x + y) = f x + f y) :
  ∀ x, f x = x :=
sorry

end NUMINAMATH_CALUDE_identity_function_characterization_l630_63089


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l630_63068

theorem cube_sum_theorem (x y z : ℝ) 
  (sum_condition : x + y + z = 5)
  (prod_sum_condition : x*y + y*z + z*x = 6)
  (prod_condition : x*y*z = -15) :
  x^3 + y^3 + z^3 = -97 := by sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l630_63068


namespace NUMINAMATH_CALUDE_equal_hot_dogs_and_buns_l630_63031

/-- The number of hot dogs in each package -/
def hot_dogs_per_package : ℕ := 7

/-- The number of buns in each package -/
def buns_per_package : ℕ := 9

/-- The smallest number of hot dog packages needed to have an equal number of hot dogs and buns -/
def smallest_number_of_packages : ℕ := 9

theorem equal_hot_dogs_and_buns :
  smallest_number_of_packages * hot_dogs_per_package =
  (smallest_number_of_packages * hot_dogs_per_package / buns_per_package) * buns_per_package ∧
  ∀ n : ℕ, n < smallest_number_of_packages →
    n * hot_dogs_per_package ≠
    (n * hot_dogs_per_package / buns_per_package) * buns_per_package :=
by sorry

end NUMINAMATH_CALUDE_equal_hot_dogs_and_buns_l630_63031


namespace NUMINAMATH_CALUDE_max_fourth_number_l630_63008

def numbers : Finset Nat := {39, 41, 44, 45, 47, 52, 55}

def is_valid_arrangement (arr : List Nat) : Prop :=
  arr.toFinset = numbers ∧
  ∀ i, i + 2 < arr.length → (arr[i]! + arr[i+1]! + arr[i+2]!) % 3 = 0

theorem max_fourth_number :
  ∃ (arr : List Nat), is_valid_arrangement arr ∧
    ∀ (other_arr : List Nat), is_valid_arrangement other_arr →
      arr[3]! ≥ other_arr[3]! ∧ arr[3]! = 47 :=
sorry

end NUMINAMATH_CALUDE_max_fourth_number_l630_63008


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l630_63072

theorem complex_modulus_equality (t : ℝ) (ht : t > 0) : 
  t = 3 * Real.sqrt 3 ↔ Complex.abs (-5 + t * Complex.I) = 2 * Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l630_63072


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l630_63067

theorem imaginary_part_of_complex_number : Complex.im (Complex.I^2 * (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l630_63067


namespace NUMINAMATH_CALUDE_red_team_score_l630_63098

theorem red_team_score (chuck_team_score : ℕ) (score_difference : ℕ) :
  chuck_team_score = 95 →
  score_difference = 19 →
  chuck_team_score - score_difference = 76 :=
by
  sorry

end NUMINAMATH_CALUDE_red_team_score_l630_63098


namespace NUMINAMATH_CALUDE_root_relation_implies_k_value_l630_63003

theorem root_relation_implies_k_value (k : ℝ) : 
  (∃ r s : ℝ, r^2 + k*r + 8 = 0 ∧ s^2 + k*s + 8 = 0 ∧
   (r+3)^2 - k*(r+3) + 8 = 0 ∧ (s+3)^2 - k*(s+3) + 8 = 0) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_relation_implies_k_value_l630_63003


namespace NUMINAMATH_CALUDE_percentage_problem_l630_63052

theorem percentage_problem (P : ℝ) : 
  (0.3 * 200 = P / 100 * 50 + 30) → P = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l630_63052


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l630_63050

/-- The line ρ cos θ + 2ρ sin θ = 1 in polar coordinates -/
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.cos θ + 2 * ρ * Real.sin θ = 1

/-- The third quadrant in Cartesian coordinates -/
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- Theorem: The line ρ cos θ + 2ρ sin θ = 1 does not pass through the third quadrant -/
theorem line_not_in_third_quadrant :
  ¬∃ (x y : ℝ), (∃ (ρ θ : ℝ), polar_line ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧ third_quadrant x y :=
sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l630_63050


namespace NUMINAMATH_CALUDE_triangle_equality_condition_l630_63015

/-- In a triangle ABC with sides a, b, and c, the equation 
    (b² * c²) / (2 * b * c * cos(A)) = b² + c² - 2 * b * c * cos(A) 
    holds if and only if a = b or a = c. -/
theorem triangle_equality_condition (a b c : ℝ) (A : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (b^2 * c^2) / (2 * b * c * Real.cos A) = b^2 + c^2 - 2 * b * c * Real.cos A ↔ 
  a = b ∨ a = c := by
sorry

end NUMINAMATH_CALUDE_triangle_equality_condition_l630_63015


namespace NUMINAMATH_CALUDE_photocopy_cost_calculation_l630_63012

/-- The cost of a single photocopy -/
def photocopy_cost : ℝ := 0.02

/-- The discount rate for large orders -/
def discount_rate : ℝ := 0.25

/-- The number of copies in a large order -/
def large_order_threshold : ℕ := 100

/-- The number of copies each person orders -/
def copies_per_person : ℕ := 80

/-- The savings per person when combining orders -/
def savings_per_person : ℝ := 0.40

theorem photocopy_cost_calculation :
  let total_copies := 2 * copies_per_person
  let undiscounted_total := total_copies * photocopy_cost
  let discounted_total := undiscounted_total * (1 - discount_rate)
  discounted_total = undiscounted_total - 2 * savings_per_person :=
by sorry

end NUMINAMATH_CALUDE_photocopy_cost_calculation_l630_63012


namespace NUMINAMATH_CALUDE_probability_A_B_different_groups_l630_63037

def number_of_people : ℕ := 6
def number_of_groups : ℕ := 3

theorem probability_A_B_different_groups :
  let total_ways := (number_of_people.choose 2) * ((number_of_people - 2).choose 2) / (number_of_groups.factorial)
  let ways_same_group := ((number_of_people - 2).choose 2) / ((number_of_groups - 1).factorial)
  (total_ways - ways_same_group) / total_ways = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_A_B_different_groups_l630_63037


namespace NUMINAMATH_CALUDE_functional_equation_solution_l630_63033

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (f y / f x + 1) = f (x + y / x + 1) - f x

/-- The main theorem stating the form of the function satisfying the equation -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x, x > 0 → f x > 0) →
  SatisfiesFunctionalEquation f →
  ∃ a : ℝ, a > 0 ∧ ∀ x, x > 0 → f x = a * x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l630_63033


namespace NUMINAMATH_CALUDE_vector_dot_product_equality_l630_63099

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-1, 3)
def C : ℝ × ℝ := (2, 1)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def vector_BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_equality : 
  dot_product vector_AB (2 • vector_AC + vector_BC) = -14 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_equality_l630_63099


namespace NUMINAMATH_CALUDE_bus_seating_problem_l630_63021

theorem bus_seating_problem :
  ∀ (bus_seats minibus_seats : ℕ),
    bus_seats = minibus_seats + 20 →
    5 * bus_seats + 5 * minibus_seats = 300 →
    bus_seats = 40 ∧ minibus_seats = 20 :=
by
  sorry

#check bus_seating_problem

end NUMINAMATH_CALUDE_bus_seating_problem_l630_63021


namespace NUMINAMATH_CALUDE_remainder_102_104_plus_6_div_9_l630_63009

theorem remainder_102_104_plus_6_div_9 : (102 * 104 + 6) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_102_104_plus_6_div_9_l630_63009


namespace NUMINAMATH_CALUDE_student_multiplication_error_l630_63004

/-- Represents a repeating decimal of the form 1.abababab... -/
def repeating_decimal (a b : ℕ) : ℚ :=
  1 + (10 * a + b : ℚ) / 99

/-- Represents the decimal 1.ab -/
def non_repeating_decimal (a b : ℕ) : ℚ :=
  1 + (a : ℚ) / 10 + (b : ℚ) / 100

theorem student_multiplication_error (a b : ℕ) :
  a < 10 → b < 10 →
  66 * (repeating_decimal a b - non_repeating_decimal a b) = (1 : ℚ) / 2 →
  a * 10 + b = 75 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_error_l630_63004


namespace NUMINAMATH_CALUDE_base5_to_octal_polynomial_evaluation_l630_63082

-- Define the base-5 number 1234₅
def base5_number : ℕ := 1 * 5^3 + 2 * 5^2 + 3 * 5^1 + 4 * 5^0

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

-- Theorem 1: Converting base-5 to octal
theorem base5_to_octal : 
  (base5_number : ℕ).digits 8 = [3, 0, 2] := by sorry

-- Theorem 2: Evaluating the polynomial at x = 3
theorem polynomial_evaluation :
  f 3 = 21324 := by sorry

end NUMINAMATH_CALUDE_base5_to_octal_polynomial_evaluation_l630_63082


namespace NUMINAMATH_CALUDE_power_2021_representation_l630_63079

theorem power_2021_representation (n : ℕ+) :
  (∃ (x y : ℤ), (2021 : ℤ)^(n : ℕ) = x^4 - 4*y^4) ↔ 4 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_power_2021_representation_l630_63079


namespace NUMINAMATH_CALUDE_simplified_tax_for_leonid_business_l630_63086

-- Define the types of tax regimes
inductive TaxRegime
  | UnifiedAgricultural
  | Simplified
  | General
  | Patent

-- Define the characteristics of a business
structure Business where
  isAgricultural : Bool
  isSmall : Bool
  hasComplexAccounting : Bool
  isNewEntrepreneur : Bool

-- Define the function to determine the appropriate tax regime
def appropriateTaxRegime (b : Business) : TaxRegime :=
  if b.isAgricultural then TaxRegime.UnifiedAgricultural
  else if b.isSmall && b.isNewEntrepreneur && !b.hasComplexAccounting then TaxRegime.Simplified
  else if !b.isSmall || b.hasComplexAccounting then TaxRegime.General
  else TaxRegime.Patent

-- Theorem statement
theorem simplified_tax_for_leonid_business :
  let leonidBusiness : Business := {
    isAgricultural := false,
    isSmall := true,
    hasComplexAccounting := false,
    isNewEntrepreneur := true
  }
  appropriateTaxRegime leonidBusiness = TaxRegime.Simplified :=
by sorry


end NUMINAMATH_CALUDE_simplified_tax_for_leonid_business_l630_63086


namespace NUMINAMATH_CALUDE_age_of_15th_person_l630_63059

theorem age_of_15th_person (total_persons : Nat) (avg_age_all : Nat) (group1_size : Nat) 
  (avg_age_group1 : Nat) (group2_size : Nat) (avg_age_group2 : Nat) :
  total_persons = 18 →
  avg_age_all = 15 →
  group1_size = 5 →
  avg_age_group1 = 14 →
  group2_size = 9 →
  avg_age_group2 = 16 →
  (total_persons * avg_age_all) = 
    (group1_size * avg_age_group1) + (group2_size * avg_age_group2) + 56 :=
by
  sorry

#check age_of_15th_person

end NUMINAMATH_CALUDE_age_of_15th_person_l630_63059


namespace NUMINAMATH_CALUDE_sum_p_q_equals_expected_p_condition_q_condition_l630_63053

/-- A linear function p(x) satisfying p(-1) = -2 -/
def p (x : ℝ) : ℝ := 4 * x - 2

/-- A quadratic function q(x) satisfying q(1) = 3 -/
def q (x : ℝ) : ℝ := 1.5 * x^2 - 1.5

/-- Theorem stating that p(x) + q(x) = 1.5x^2 + 4x - 3.5 -/
theorem sum_p_q_equals_expected : 
  ∀ x : ℝ, p x + q x = 1.5 * x^2 + 4 * x - 3.5 := by
  sorry

/-- Verification of the conditions -/
theorem p_condition : p (-1) = -2 := by
  sorry

theorem q_condition : q 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_p_q_equals_expected_p_condition_q_condition_l630_63053


namespace NUMINAMATH_CALUDE_triangle_angles_l630_63047

-- Define a triangle XYZ
structure Triangle :=
  (X Y Z : Point)

-- Define the angles in the triangle
def angle_YXZ (t : Triangle) : ℝ := sorry
def angle_XYZ (t : Triangle) : ℝ := sorry
def angle_XZY (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_angles (t : Triangle) :
  angle_YXZ t = 40 ∧ angle_XYZ t = 80 → angle_XZY t = 60 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_l630_63047


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l630_63000

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x

-- State the theorem
theorem monotonic_increasing_condition (a : ℝ) : 
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, Monotone (fun x => f a x)) → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l630_63000


namespace NUMINAMATH_CALUDE_x_less_equal_two_l630_63088

theorem x_less_equal_two (x : ℝ) (h : Real.sqrt ((x - 2)^2) = 2 - x) : x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_x_less_equal_two_l630_63088


namespace NUMINAMATH_CALUDE_solution_set_l630_63034

/-- A linear function f(x) = ax + b where a > 0 and f(-2) = 0 -/
def f (a b : ℝ) (ha : a > 0) (hf : a * (-2) + b = 0) (x : ℝ) : ℝ :=
  a * x + b

theorem solution_set (a b x : ℝ) (ha : a > 0) (hf : a * (-2) + b = 0) :
  a * x > b ↔ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l630_63034


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l630_63032

theorem quadratic_form_minimum : ∀ x y : ℝ,
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 7 ≥ 28 ∧
  ∃ x y : ℝ, 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 7 = 28 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l630_63032


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l630_63085

/-- Represents the composition of a cricket team -/
structure CricketTeam where
  total_players : ℕ
  throwers : ℕ
  hitters : ℕ
  runners : ℕ
  left_handed_hitters : ℕ
  left_handed_runners : ℕ

/-- Calculates the total number of right-handed players in a cricket team -/
def right_handed_players (team : CricketTeam) : ℕ :=
  team.throwers + (team.hitters - team.left_handed_hitters) + (team.runners - team.left_handed_runners)

/-- Theorem stating the total number of right-handed players in the given cricket team -/
theorem cricket_team_right_handed_players :
  ∃ (team : CricketTeam),
    team.total_players = 300 ∧
    team.throwers = 165 ∧
    team.hitters = team.runners ∧
    team.hitters + team.runners = team.total_players - team.throwers ∧
    team.left_handed_hitters * 5 = team.hitters * 2 ∧
    team.left_handed_runners * 7 = team.runners * 3 ∧
    right_handed_players team = 243 :=
  sorry


end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l630_63085


namespace NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l630_63040

theorem three_greater_than_sqrt_seven : 3 > Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l630_63040


namespace NUMINAMATH_CALUDE_complementary_events_l630_63081

-- Define the sample space for throwing 3 coins
def CoinOutcome := Fin 2 × Fin 2 × Fin 2

-- Define the event "No more than one head"
def NoMoreThanOneHead (outcome : CoinOutcome) : Prop :=
  (outcome.1 + outcome.2.1 + outcome.2.2 : ℕ) ≤ 1

-- Define the event "At least two heads"
def AtLeastTwoHeads (outcome : CoinOutcome) : Prop :=
  (outcome.1 + outcome.2.1 + outcome.2.2 : ℕ) ≥ 2

-- Theorem stating that the two events are complementary
theorem complementary_events :
  ∀ (outcome : CoinOutcome), NoMoreThanOneHead outcome ↔ ¬(AtLeastTwoHeads outcome) :=
by
  sorry


end NUMINAMATH_CALUDE_complementary_events_l630_63081


namespace NUMINAMATH_CALUDE_cyclic_fourth_root_sum_inequality_l630_63061

theorem cyclic_fourth_root_sum_inequality (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (ha₁ : a₁ > 0) (ha₂ : a₂ > 0) (ha₃ : a₃ > 0) (ha₄ : a₄ > 0) (ha₅ : a₅ > 0) (ha₆ : a₆ > 0) : 
  (a₁ / (a₂ + a₃ + a₄)) ^ (1/4 : ℝ) + 
  (a₂ / (a₃ + a₄ + a₅)) ^ (1/4 : ℝ) + 
  (a₃ / (a₄ + a₅ + a₆)) ^ (1/4 : ℝ) + 
  (a₄ / (a₅ + a₆ + a₁)) ^ (1/4 : ℝ) + 
  (a₅ / (a₆ + a₁ + a₂)) ^ (1/4 : ℝ) + 
  (a₆ / (a₁ + a₂ + a₃)) ^ (1/4 : ℝ) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_fourth_root_sum_inequality_l630_63061


namespace NUMINAMATH_CALUDE_geometric_sum_divisors_l630_63058

/-- The sum of geometric series from 0 to n with ratio a -/
def geometric_sum (a : ℕ) (n : ℕ) : ℕ :=
  (a^(n+1) - 1) / (a - 1)

/-- The set of all divisors of geometric_sum a n for some n -/
def divisor_set (a : ℕ) : Set ℕ :=
  {m : ℕ | ∃ n : ℕ, (geometric_sum a n) % m = 0}

/-- The set of all natural numbers relatively prime to a -/
def coprime_set (a : ℕ) : Set ℕ :=
  {m : ℕ | Nat.gcd m a = 1}

theorem geometric_sum_divisors (a : ℕ) (h : a > 1) :
  divisor_set a = coprime_set a :=
by sorry

end NUMINAMATH_CALUDE_geometric_sum_divisors_l630_63058


namespace NUMINAMATH_CALUDE_roots_form_triangle_l630_63090

/-- The roots of the equation (x-1)(x^2-2x+m) = 0 can form a triangle if and only if 3/4 < m ≤ 1 -/
theorem roots_form_triangle (m : ℝ) : 
  (∃ a b c : ℝ, 
    (a - 1) * (a^2 - 2*a + m) = 0 ∧
    (b - 1) * (b^2 - 2*b + m) = 0 ∧
    (c - 1) * (c^2 - 2*c + m) = 0 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ c + a > b) ↔
  (3/4 < m ∧ m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_roots_form_triangle_l630_63090


namespace NUMINAMATH_CALUDE_not_all_prime_in_sequence_l630_63084

-- Define the recursive sequence
def x (n : ℕ) (x₀ a b : ℕ) : ℕ :=
  match n with
  | 0 => x₀
  | n + 1 => x n x₀ a b * a + b

-- Theorem statement
theorem not_all_prime_in_sequence (x₀ a b : ℕ) :
  ∃ n : ℕ, ¬ Nat.Prime (x n x₀ a b) :=
by sorry

end NUMINAMATH_CALUDE_not_all_prime_in_sequence_l630_63084


namespace NUMINAMATH_CALUDE_expression_evaluation_l630_63065

theorem expression_evaluation :
  let x : ℝ := -3
  let numerator := 5 + x * (5 + x) - 5^2
  let denominator := x - 5 + x^2
  numerator / denominator = -26 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l630_63065


namespace NUMINAMATH_CALUDE_inequality_proof_l630_63055

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / (a + b)) * ((a + 2*b) / (a + 3*b)) < Real.sqrt (a / (a + 4*b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l630_63055


namespace NUMINAMATH_CALUDE_wage_increase_l630_63071

/-- Represents the regression equation for monthly wage based on labor productivity -/
def wage_equation (x : ℝ) : ℝ := 50 + 60 * x

/-- Theorem stating that an increase of 1 in labor productivity results in a 60 yuan increase in monthly wage -/
theorem wage_increase (x : ℝ) : wage_equation (x + 1) = wage_equation x + 60 := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_l630_63071


namespace NUMINAMATH_CALUDE_characterize_bijection_condition_l630_63013

/-- Given an even positive integer m, characterize all positive integers n for which
    there exists a bijection f from [1,n] to [1,n] satisfying the condition that
    for all x and y in [1,n] where n divides mx - y, n+1 divides f(x)^m - f(y). -/
theorem characterize_bijection_condition (m : ℕ) (h_m : Even m) (h_m_pos : 0 < m) :
  ∀ n : ℕ, 0 < n →
    (∃ f : Fin n → Fin n, Function.Bijective f ∧
      ∀ x y : Fin n, n ∣ m * x - y →
        (n + 1) ∣ (f x)^m - (f y)) ↔
    Nat.Prime (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_characterize_bijection_condition_l630_63013


namespace NUMINAMATH_CALUDE_gretchen_earnings_l630_63087

/-- Gretchen's caricature business --/
def caricature_problem (price_per_drawing : ℚ) (saturday_sales : ℕ) (sunday_sales : ℕ) : ℚ :=
  (saturday_sales + sunday_sales : ℚ) * price_per_drawing

/-- Theorem stating the total money Gretchen made --/
theorem gretchen_earnings :
  caricature_problem 20 24 16 = 800 := by
  sorry

end NUMINAMATH_CALUDE_gretchen_earnings_l630_63087


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l630_63043

theorem quadratic_root_problem (a : ℝ) :
  ((-1 : ℝ)^2 - 2*(-1) + a = 0) → 
  (∃ x : ℝ, x^2 - 2*x + a = 0 ∧ x ≠ -1 ∧ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l630_63043


namespace NUMINAMATH_CALUDE_negation_of_forall_ge_two_l630_63035

theorem negation_of_forall_ge_two :
  (¬ (∀ x : ℝ, x > 0 → x + 1/x ≥ 2)) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_ge_two_l630_63035


namespace NUMINAMATH_CALUDE_percent_of_percent_l630_63048

theorem percent_of_percent (x : ℝ) (h : x ≠ 0) : (0.3 * 0.7 * x) / x = 0.21 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l630_63048


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l630_63011

theorem simplify_nested_expression (x : ℝ) : 1 - (1 + (1 - (1 + (1 - x)))) = 1 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l630_63011


namespace NUMINAMATH_CALUDE_equation_system_solution_l630_63064

theorem equation_system_solution (a b x y : ℝ) 
  (h1 : x^2 + x*y + y^2 = a^2)
  (h2 : Real.log (Real.sqrt a) / Real.log (a^(1/x)) + Real.log (Real.sqrt b) / Real.log (b^(1/y)) = a / Real.sqrt 3) :
  x = a * Real.sqrt 3 / 3 ∧ y = a * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_equation_system_solution_l630_63064


namespace NUMINAMATH_CALUDE_max_value_implies_a_l630_63001

theorem max_value_implies_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, 2 * a^x - 5 ≤ 10) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, 2 * a^x - 5 = 10) →
  a = Real.sqrt (15/2) ∨ a = 15/2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l630_63001


namespace NUMINAMATH_CALUDE_full_time_employees_count_l630_63017

/-- A corporation with part-time and full-time employees -/
structure Corporation where
  total_employees : ℕ
  part_time_employees : ℕ

/-- The number of full-time employees in a corporation -/
def full_time_employees (c : Corporation) : ℕ :=
  c.total_employees - c.part_time_employees

/-- Theorem stating the number of full-time employees in a specific corporation -/
theorem full_time_employees_count (c : Corporation) 
  (h1 : c.total_employees = 65134)
  (h2 : c.part_time_employees = 2041) :
  full_time_employees c = 63093 := by
  sorry

end NUMINAMATH_CALUDE_full_time_employees_count_l630_63017


namespace NUMINAMATH_CALUDE_quadratic_max_value_l630_63083

/-- The quadratic function f(x) = -x^2 + 2x has a maximum value of 1. -/
theorem quadratic_max_value (x : ℝ) : 
  (∀ y : ℝ, -y^2 + 2*y ≤ 1) ∧ (∃ z : ℝ, -z^2 + 2*z = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l630_63083


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l630_63007

/-- Properties of a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ
  angle_F1PF2 : ℝ
  area_F1PF2 : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  e_eq : e = 2
  angle_eq : angle_F1PF2 = Real.pi / 2
  area_eq : area_F1PF2 = 3

/-- The length of the real axis of a hyperbola -/
def real_axis_length (h : Hyperbola) : ℝ := 2 * h.a

/-- Theorem: The length of the real axis of the given hyperbola is 2 -/
theorem hyperbola_real_axis_length (h : Hyperbola) : real_axis_length h = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l630_63007


namespace NUMINAMATH_CALUDE_initial_machines_count_l630_63038

/-- The number of machines initially operating to fill a production order -/
def initial_machines : ℕ := sorry

/-- The total number of machines available -/
def total_machines : ℕ := 7

/-- The time taken by the initial number of machines to fill the order (in hours) -/
def initial_time : ℕ := 42

/-- The time taken by all machines to fill the order (in hours) -/
def all_machines_time : ℕ := 36

/-- The rate at which each machine works (assumed to be constant and positive) -/
def machine_rate : ℝ := sorry

theorem initial_machines_count :
  initial_machines = 6 :=
by sorry

end NUMINAMATH_CALUDE_initial_machines_count_l630_63038


namespace NUMINAMATH_CALUDE_quadratic_function_unique_a_l630_63051

/-- Represents a quadratic function of the form f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluates the quadratic function at a given x -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_function_unique_a (f : QuadraticFunction) :
  f.eval 1 = 5 → f.eval 0 = 2 → f.a = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_a_l630_63051


namespace NUMINAMATH_CALUDE_solution_set_implies_a_values_l630_63046

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- State the theorem
theorem solution_set_implies_a_values :
  (∀ x : ℝ, f a x ≤ 0 ↔ x ≤ -1) →
  (a = 2 ∨ a = -4) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_values_l630_63046


namespace NUMINAMATH_CALUDE_unique_pizza_combinations_l630_63042

def num_toppings : ℕ := 8
def toppings_per_pizza : ℕ := 3

theorem unique_pizza_combinations :
  Nat.choose num_toppings toppings_per_pizza = 56 := by
  sorry

end NUMINAMATH_CALUDE_unique_pizza_combinations_l630_63042


namespace NUMINAMATH_CALUDE_feuerbach_circle_equation_l630_63018

/-- The Feuerbach circle (nine-point circle) of a triangle -/
def feuerbach_circle (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * c * (p.1^2 + p.2^2) - (a + b) * c * p.1 + (a * b - c^2) * p.2 = 0}

/-- The vertices of the triangle -/
def triangle_vertices (a b c : ℝ) : Set (ℝ × ℝ) :=
  {(a, 0), (b, 0), (0, c)}

theorem feuerbach_circle_equation (a b c : ℝ) (h : c ≠ 0) :
  ∃ (circle : Set (ℝ × ℝ)), circle = feuerbach_circle a b c ∧
  (∀ (p : ℝ × ℝ), p ∈ circle ↔ 2 * c * (p.1^2 + p.2^2) - (a + b) * c * p.1 + (a * b - c^2) * p.2 = 0) :=
sorry

end NUMINAMATH_CALUDE_feuerbach_circle_equation_l630_63018


namespace NUMINAMATH_CALUDE_students_in_all_classes_l630_63024

/-- Represents the number of students registered for a combination of classes -/
structure ClassRegistration where
  history : ℕ
  math : ℕ
  english : ℕ
  historyMath : ℕ
  historyEnglish : ℕ
  mathEnglish : ℕ
  allThree : ℕ

/-- The theorem stating the number of students registered for all three classes -/
theorem students_in_all_classes 
  (total : ℕ) 
  (classes : ClassRegistration) 
  (h1 : total = 86)
  (h2 : classes.history = 12)
  (h3 : classes.math = 17)
  (h4 : classes.english = 36)
  (h5 : classes.historyMath + classes.historyEnglish + classes.mathEnglish = 3)
  (h6 : total = classes.history + classes.math + classes.english - 
        (classes.historyMath + classes.historyEnglish + classes.mathEnglish) + 
        classes.allThree) :
  classes.allThree = 24 := by
  sorry

end NUMINAMATH_CALUDE_students_in_all_classes_l630_63024


namespace NUMINAMATH_CALUDE_parakeets_per_cage_l630_63095

/-- Given a pet store with bird cages, prove the number of parakeets in each cage -/
theorem parakeets_per_cage 
  (num_cages : ℕ) 
  (parrots_per_cage : ℕ) 
  (total_birds : ℕ) 
  (h1 : num_cages = 9)
  (h2 : parrots_per_cage = 2)
  (h3 : total_birds = 72) :
  (total_birds - num_cages * parrots_per_cage) / num_cages = 6 := by
  sorry

end NUMINAMATH_CALUDE_parakeets_per_cage_l630_63095


namespace NUMINAMATH_CALUDE_max_performances_l630_63075

/-- Represents a performance in the theater festival -/
structure Performance :=
  (students : Finset ℕ)
  (size_eq_six : students.card = 6)

/-- The theater festival -/
structure TheaterFestival :=
  (num_students : ℕ)
  (num_students_eq_twelve : num_students = 12)
  (performances : Finset Performance)
  (common_students : Performance → Performance → Finset ℕ)
  (common_students_le_two : ∀ p1 p2 : Performance, p1 ≠ p2 → (common_students p1 p2).card ≤ 2)

/-- The theorem stating the maximum number of performances -/
theorem max_performances (festival : TheaterFestival) : 
  festival.performances.card ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_performances_l630_63075


namespace NUMINAMATH_CALUDE_even_function_symmetric_about_y_axis_l630_63030

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_symmetric_about_y_axis (f : ℝ → ℝ) (h : even_function f) :
  ∀ x y, f x = y ↔ f (-x) = y :=
sorry

end NUMINAMATH_CALUDE_even_function_symmetric_about_y_axis_l630_63030


namespace NUMINAMATH_CALUDE_arithmetic_sequence_zero_term_l630_63096

/-- An arithmetic sequence with common difference d ≠ 0 -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  h : d ≠ 0  -- d is non-zero
  seq : ∀ n : ℕ, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- The theorem statement -/
theorem arithmetic_sequence_zero_term
  (seq : ArithmeticSequence)
  (h : seq.a 3 + seq.a 9 = seq.a 10 - seq.a 8) :
  ∃! n : ℕ, seq.a n = 0 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_zero_term_l630_63096


namespace NUMINAMATH_CALUDE_sum_of_specific_arithmetic_progression_l630_63091

/-- Sum of an arithmetic progression -/
def sum_arithmetic_progression (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Theorem: The sum of the first 20 terms of an arithmetic progression
    with first term 30 and common difference -3 is equal to 30 -/
theorem sum_of_specific_arithmetic_progression :
  sum_arithmetic_progression 30 (-3) 20 = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_specific_arithmetic_progression_l630_63091


namespace NUMINAMATH_CALUDE_geometric_sequence_terms_l630_63049

/-- Given a geometric sequence with third term 12 and fourth term 18, prove that the first term is 16/3 and the second term is 8. -/
theorem geometric_sequence_terms (a : ℕ → ℚ) (q : ℚ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 3 = 12 →                    -- Third term is 12
  a 4 = 18 →                    -- Fourth term is 18
  a 1 = 16 / 3 ∧ a 2 = 8 :=     -- First term is 16/3 and second term is 8
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_terms_l630_63049


namespace NUMINAMATH_CALUDE_percentage_problem_l630_63020

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 1000) : 1.20 * x = 6000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l630_63020


namespace NUMINAMATH_CALUDE_chord_length_l630_63014

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l630_63014


namespace NUMINAMATH_CALUDE_total_games_won_l630_63016

def team_games_won (games_played : ℕ) (win_percentage : ℚ) : ℕ :=
  ⌊(games_played : ℚ) * win_percentage⌋₊

theorem total_games_won :
  let team_a := team_games_won 150 (35/100)
  let team_b := team_games_won 110 (45/100)
  let team_c := team_games_won 200 (30/100)
  team_a + team_b + team_c = 163 := by
  sorry

end NUMINAMATH_CALUDE_total_games_won_l630_63016


namespace NUMINAMATH_CALUDE_claire_took_eight_photos_l630_63070

/-- The number of photos taken by Claire -/
def claire_photos : ℕ := 8

/-- The number of photos taken by Lisa -/
def lisa_photos : ℕ := 3 * claire_photos

/-- The number of photos taken by Robert -/
def robert_photos : ℕ := claire_photos + 16

/-- Theorem stating that given the conditions, Claire has taken 8 photos -/
theorem claire_took_eight_photos :
  lisa_photos = robert_photos ∧
  lisa_photos = 3 * claire_photos ∧
  robert_photos = claire_photos + 16 →
  claire_photos = 8 := by
  sorry

end NUMINAMATH_CALUDE_claire_took_eight_photos_l630_63070


namespace NUMINAMATH_CALUDE_tower_height_l630_63097

/-- The height of a tower given specific angle measurements -/
theorem tower_height (angle1 angle2 : Real) (distance : Real) (height : Real) : 
  angle1 = Real.pi / 6 →  -- 30 degrees in radians
  angle2 = Real.pi / 4 →  -- 45 degrees in radians
  distance = 20 → 
  Real.tan angle1 = height / (height + distance) →
  height = 10 * (Real.sqrt 3 + 1) :=
by sorry

end NUMINAMATH_CALUDE_tower_height_l630_63097


namespace NUMINAMATH_CALUDE_point_plane_line_sphere_ratio_l630_63027

/-- Given a point (a,b,c) on a plane and a line through the origin, 
    and (p,q,r) as the center of a sphere passing through specific points,
    prove that (a+b+c)/(p+q+r) = 1 -/
theorem point_plane_line_sphere_ratio 
  (a b c d e f p q r : ℝ) 
  (h1 : ∃ (t : ℝ), a = t * d ∧ b = t * e ∧ c = t * f)  -- (a,b,c) on line with direction (d,e,f)
  (h2 : ∃ (α β γ : ℝ), α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0 ∧        -- A, B, C distinct from O
        a/α + b/β + c/γ = 1)                          -- (a,b,c) on plane through A, B, C
  (h3 : p^2 + q^2 + r^2 = (p - α)^2 + q^2 + r^2)       -- O and A equidistant from (p,q,r)
  (h4 : p^2 + q^2 + r^2 = p^2 + (q - β)^2 + r^2)       -- O and B equidistant from (p,q,r)
  (h5 : p^2 + q^2 + r^2 = p^2 + q^2 + (r - γ)^2)       -- O and C equidistant from (p,q,r)
  (h6 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)                        -- Avoid division by zero
  : (a + b + c) / (p + q + r) = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_plane_line_sphere_ratio_l630_63027


namespace NUMINAMATH_CALUDE_m_range_l630_63056

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∃ x : ℝ, 3^x - m + 1 ≤ 0

-- Define the theorem
theorem m_range :
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  (∀ m : ℝ, 1 < m ∧ m ≤ 2 ↔ q m ∧ ¬(p m)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l630_63056


namespace NUMINAMATH_CALUDE_absolute_value_two_l630_63074

theorem absolute_value_two (m : ℝ) : |m| = 2 → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_two_l630_63074


namespace NUMINAMATH_CALUDE_min_value_x_plus_reciprocal_l630_63019

theorem min_value_x_plus_reciprocal (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_reciprocal_l630_63019


namespace NUMINAMATH_CALUDE_largest_sum_is_ten_l630_63028

/-- A structure representing a set of five positive integers -/
structure FiveIntegers where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+

/-- The property that the sum of the five integers equals their product -/
def hasSumProductProperty (x : FiveIntegers) : Prop :=
  x.a + x.b + x.c + x.d + x.e = x.a * x.b * x.c * x.d * x.e

/-- The sum of the five integers -/
def sum (x : FiveIntegers) : ℕ :=
  x.a + x.b + x.c + x.d + x.e

/-- The theorem stating that (1, 1, 1, 2, 5) has the largest sum among all valid sets -/
theorem largest_sum_is_ten :
  ∀ x : FiveIntegers, hasSumProductProperty x → sum x ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_largest_sum_is_ten_l630_63028


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l630_63045

theorem simplify_sqrt_expression : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 392 / Real.sqrt 98) = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l630_63045


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l630_63022

/-- Proves that given a boat with a speed of 20 km/hr in still water,
if it travels 26 km downstream and 14 km upstream in the same time,
then the speed of the stream is 6 km/hr. -/
theorem stream_speed_calculation (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) :
  boat_speed = 20 →
  downstream_distance = 26 →
  upstream_distance = 14 →
  (downstream_distance / (boat_speed + x) = upstream_distance / (boat_speed - x)) →
  x = 6 :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_calculation_l630_63022


namespace NUMINAMATH_CALUDE_room_volume_example_l630_63041

/-- The volume of a rectangular room -/
def room_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a room with dimensions 100 m * 10 m * 10 m is 10,000 cubic meters -/
theorem room_volume_example : room_volume 100 10 10 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_room_volume_example_l630_63041


namespace NUMINAMATH_CALUDE_tip_percentage_calculation_l630_63077

theorem tip_percentage_calculation (meal_cost drink_cost payment change : ℚ) : 
  meal_cost = 10 →
  drink_cost = 5/2 →
  payment = 20 →
  change = 5 →
  ((payment - change) - (meal_cost + drink_cost)) / (meal_cost + drink_cost) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_calculation_l630_63077


namespace NUMINAMATH_CALUDE_max_points_in_tournament_l630_63069

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (games_per_pair : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Calculates the total number of games in the tournament -/
def total_games (t : Tournament) : ℕ :=
  t.num_teams.choose 2 * t.games_per_pair

/-- Calculates the total points available in the tournament -/
def total_points (t : Tournament) : ℕ :=
  total_games t * t.points_for_win

/-- Represents the maximum points achievable by top teams -/
def max_points_for_top_teams (t : Tournament) : ℕ :=
  let points_from_top_matches := (t.num_teams - 1) * t.points_for_win
  let points_from_other_matches := (t.num_teams - 3) * 2 * t.points_for_win
  points_from_top_matches + points_from_other_matches

/-- The main theorem to be proved -/
theorem max_points_in_tournament (t : Tournament) 
  (h1 : t.num_teams = 8)
  (h2 : t.games_per_pair = 2)
  (h3 : t.points_for_win = 3)
  (h4 : t.points_for_draw = 1)
  (h5 : t.points_for_loss = 0) :
  max_points_for_top_teams t = 38 :=
sorry

end NUMINAMATH_CALUDE_max_points_in_tournament_l630_63069


namespace NUMINAMATH_CALUDE_extremum_value_l630_63039

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Define the derivative of f(x)
def f_prime (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Theorem statement
theorem extremum_value (a b : ℝ) : 
  f a b 1 = 10 ∧ f_prime a b 1 = 0 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_extremum_value_l630_63039


namespace NUMINAMATH_CALUDE_house_rent_expenditure_l630_63092

-- Define the given parameters
def total_income : ℝ := 1000
def petrol_percentage : ℝ := 0.30
def house_rent_percentage : ℝ := 0.10
def petrol_expenditure : ℝ := 300

-- Define the theorem
theorem house_rent_expenditure :
  let remaining_income := total_income - petrol_expenditure
  remaining_income * house_rent_percentage = 70 := by
  sorry

end NUMINAMATH_CALUDE_house_rent_expenditure_l630_63092


namespace NUMINAMATH_CALUDE_total_weekly_meals_l630_63010

/-- The number of days in a week -/
def daysInWeek : ℕ := 7

/-- The number of meals served daily by the first restaurant -/
def restaurant1Meals : ℕ := 20

/-- The number of meals served daily by the second restaurant -/
def restaurant2Meals : ℕ := 40

/-- The number of meals served daily by the third restaurant -/
def restaurant3Meals : ℕ := 50

/-- Theorem stating that the total number of meals served per week by the three restaurants is 770 -/
theorem total_weekly_meals :
  (restaurant1Meals * daysInWeek) + (restaurant2Meals * daysInWeek) + (restaurant3Meals * daysInWeek) = 770 := by
  sorry

end NUMINAMATH_CALUDE_total_weekly_meals_l630_63010


namespace NUMINAMATH_CALUDE_grade_distribution_l630_63036

theorem grade_distribution (thompson_total : ℕ) (thompson_a : ℕ) (thompson_b : ℕ) (carter_total : ℕ)
  (h1 : thompson_total = 20)
  (h2 : thompson_a = 12)
  (h3 : thompson_b = 5)
  (h4 : carter_total = 30)
  (h5 : thompson_a + thompson_b ≤ thompson_total) :
  ∃ (carter_a carter_b : ℕ),
    carter_a + carter_b ≤ carter_total ∧
    carter_a * thompson_total = thompson_a * carter_total ∧
    carter_b * (thompson_total - thompson_a) = thompson_b * (carter_total - carter_a) ∧
    carter_a = 18 ∧
    carter_b = 8 :=
by sorry

end NUMINAMATH_CALUDE_grade_distribution_l630_63036


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_min_value_fraction_sum_achievable_l630_63073

theorem min_value_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (8 / x + 2 / y) ≥ 18 :=
by sorry

theorem min_value_fraction_sum_achievable :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ (8 / x + 2 / y) = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_min_value_fraction_sum_achievable_l630_63073


namespace NUMINAMATH_CALUDE_twice_a_minus_four_nonnegative_l630_63006

theorem twice_a_minus_four_nonnegative (a : ℝ) :
  (2 * a - 4 ≥ 0) ↔ (∃ (x : ℝ), x ≥ 0 ∧ x = 2 * a - 4) :=
by sorry

end NUMINAMATH_CALUDE_twice_a_minus_four_nonnegative_l630_63006
