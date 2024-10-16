import Mathlib

namespace NUMINAMATH_CALUDE_trajectory_of_M_l1285_128518

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the slope sum condition
def slope_sum_condition (x y : ℝ) : Prop :=
  y / (x + 2) + y / (x - 2) = 2

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop :=
  x * y - x^2 + 4 = 0

-- Theorem statement
theorem trajectory_of_M (x y : ℝ) (h1 : y ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ -2) :
  slope_sum_condition x y → trajectory_equation x y :=
by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_M_l1285_128518


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l1285_128538

theorem volleyball_team_selection (n : ℕ) (k : ℕ) (t : ℕ) :
  n = 15 →
  k = 6 →
  t = 3 →
  (Nat.choose n k) - (Nat.choose (n - t) k) = 4081 :=
by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l1285_128538


namespace NUMINAMATH_CALUDE_correct_mark_is_ten_l1285_128599

/-- Proves that the correct mark of a student is 10, given the conditions of the problem. -/
theorem correct_mark_is_ten (n : ℕ) (initial_avg final_avg wrong_mark : ℚ) :
  n = 30 →
  initial_avg = 100 →
  wrong_mark = 70 →
  final_avg = 98 →
  (n : ℚ) * initial_avg - wrong_mark + (n : ℚ) * final_avg = (n : ℚ) * initial_avg →
  (n : ℚ) * initial_avg - wrong_mark + 10 = (n : ℚ) * final_avg :=
by sorry

end NUMINAMATH_CALUDE_correct_mark_is_ten_l1285_128599


namespace NUMINAMATH_CALUDE_student_divisor_problem_l1285_128533

theorem student_divisor_problem (dividend : ℕ) (student_answer : ℕ) (correct_answer : ℕ) (correct_divisor : ℕ) : 
  student_answer = 24 →
  correct_answer = 32 →
  correct_divisor = 36 →
  dividend / correct_divisor = correct_answer →
  ∃ (student_divisor : ℕ), 
    dividend / student_divisor = student_answer ∧ 
    student_divisor = 48 :=
by sorry

end NUMINAMATH_CALUDE_student_divisor_problem_l1285_128533


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1285_128509

theorem sin_2alpha_value (α : Real) (h : Real.tan (π/4 + α) = 2) : 
  Real.sin (2 * α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1285_128509


namespace NUMINAMATH_CALUDE_stock_price_change_l1285_128543

theorem stock_price_change (total_stocks : ℕ) (higher_percentage : ℚ) :
  total_stocks = 1980 →
  higher_percentage = 120 / 100 →
  ∃ (higher lower : ℕ),
    higher + lower = total_stocks ∧
    higher = (higher_percentage * lower).num ∧
    higher = 1080 :=
by sorry

end NUMINAMATH_CALUDE_stock_price_change_l1285_128543


namespace NUMINAMATH_CALUDE_count_valid_m_l1285_128542

def is_valid (m : ℕ+) : Prop :=
  ∃ k : ℕ+, (2310 : ℚ) / ((m : ℚ)^2 - 2) = k

theorem count_valid_m :
  ∃! (s : Finset ℕ+), s.card = 3 ∧ ∀ m : ℕ+, m ∈ s ↔ is_valid m :=
sorry

end NUMINAMATH_CALUDE_count_valid_m_l1285_128542


namespace NUMINAMATH_CALUDE_largest_common_divisor_l1285_128523

theorem largest_common_divisor : 
  ∃ (n : ℕ), n = 35 ∧ 
  n ∣ 420 ∧ n ∣ 385 ∧ 
  ∀ (m : ℕ), m ∣ 420 ∧ m ∣ 385 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_l1285_128523


namespace NUMINAMATH_CALUDE_fraction_equality_l1285_128527

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3/4) :
  (a - c) * (b - d) / ((a - b) * (c - d)) = -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1285_128527


namespace NUMINAMATH_CALUDE_basketball_score_proof_l1285_128575

theorem basketball_score_proof (two_points three_points free_throws : ℕ) : 
  (3 * three_points = 2 * two_points) →
  (free_throws = 2 * three_points) →
  (2 * two_points + 3 * three_points + free_throws = 72) →
  free_throws = 18 := by
sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l1285_128575


namespace NUMINAMATH_CALUDE_inverse_odd_implies_a_eq_one_l1285_128591

/-- A function f: ℝ → ℝ -/
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x ↦ 2^x - a * 2^(-x)

/-- The inverse function of f -/
noncomputable def f_inv (a : ℝ) : ℝ → ℝ := Function.invFun (f a)

/-- Theorem stating that if f_inv is odd and a is positive, then a = 1 -/
theorem inverse_odd_implies_a_eq_one (a : ℝ) (h_pos : a > 0) 
  (h_odd : ∀ x, f_inv a (-x) = -(f_inv a x)) : a = 1 := by
  sorry

#check inverse_odd_implies_a_eq_one

end NUMINAMATH_CALUDE_inverse_odd_implies_a_eq_one_l1285_128591


namespace NUMINAMATH_CALUDE_calculate_second_oil_price_l1285_128521

/-- Given a mixture of two oils, calculate the price of the second oil -/
theorem calculate_second_oil_price (volume1 volume2 price1 price_mixture : ℝ) 
  (h1 : volume1 = 10)
  (h2 : volume2 = 5)
  (h3 : price1 = 55)
  (h4 : price_mixture = 58.67) : 
  ∃ (price2 : ℝ), price2 = 66.01 ∧ 
  (volume1 * price1 + volume2 * price2) / (volume1 + volume2) = price_mixture := by
  sorry

#check calculate_second_oil_price

end NUMINAMATH_CALUDE_calculate_second_oil_price_l1285_128521


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1285_128588

theorem geometric_sequence_sixth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^7 = 2) : 
  a * r^5 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1285_128588


namespace NUMINAMATH_CALUDE_unique_solution_congruences_l1285_128578

theorem unique_solution_congruences :
  ∃! x : ℕ, x < 120 ∧
    (4 + x) % 8 = 3^2 % 8 ∧
    (6 + x) % 27 = 4^2 % 27 ∧
    (8 + x) % 125 = 6^2 % 125 ∧
    x = 37 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_congruences_l1285_128578


namespace NUMINAMATH_CALUDE_range_of_a_l1285_128558

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → |x + 1/x| > |a - 5| + 1) → 
  a ∈ Set.Ioo 4 6 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1285_128558


namespace NUMINAMATH_CALUDE_john_taxes_l1285_128540

/-- Calculate the total tax given a progressive tax system and taxable income -/
def calculate_tax (taxable_income : ℕ) : ℕ :=
  let tax1 := min taxable_income 20000 * 10 / 100
  let tax2 := min (max (taxable_income - 20000) 0) 30000 * 15 / 100
  let tax3 := min (max (taxable_income - 50000) 0) 50000 * 20 / 100
  let tax4 := max (taxable_income - 100000) 0 * 25 / 100
  tax1 + tax2 + tax3 + tax4

/-- John's financial situation -/
theorem john_taxes :
  let main_job := 75000
  let freelance := 25000
  let rental := 15000
  let dividends := 10000
  let mortgage_deduction := 32000
  let retirement_deduction := 15000
  let charitable_deduction := 10000
  let education_credit := 3000
  let total_income := main_job + freelance + rental + dividends
  let total_deductions := mortgage_deduction + retirement_deduction + charitable_deduction + education_credit
  let taxable_income := total_income - total_deductions
  taxable_income = 65000 ∧ calculate_tax taxable_income = 9500 := by
  sorry


end NUMINAMATH_CALUDE_john_taxes_l1285_128540


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1285_128548

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the non-collinearity condition
variable (h_non_collinear : ¬ ∃ (c : ℝ), e₁ = c • e₂)

-- Define vectors a and b
def a (e₁ e₂ : V) : V := 2 • e₁ + 3 • e₂
def b (k : ℝ) (e₁ e₂ : V) : V := k • e₁ - 4 • e₂

-- State the theorem
theorem parallel_vectors_k_value 
  (h_parallel : ∃ (m : ℝ), a e₁ e₂ = m • (b k e₁ e₂)) :
  k = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1285_128548


namespace NUMINAMATH_CALUDE_smallest_multiple_of_5_and_711_l1285_128547

theorem smallest_multiple_of_5_and_711 : 
  ∀ n : ℕ, n > 0 ∧ 5 ∣ n ∧ 711 ∣ n → n ≥ 3555 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_5_and_711_l1285_128547


namespace NUMINAMATH_CALUDE_curtain_cost_calculation_l1285_128525

/-- The cost of each pair of curtains -/
def curtain_cost : ℝ := 30

/-- The number of curtain pairs purchased -/
def curtain_pairs : ℕ := 2

/-- The number of wall prints purchased -/
def wall_prints : ℕ := 9

/-- The cost of each wall print -/
def wall_print_cost : ℝ := 15

/-- The cost of installation service -/
def installation_cost : ℝ := 50

/-- The total cost of the purchase -/
def total_cost : ℝ := 245

theorem curtain_cost_calculation :
  curtain_cost * curtain_pairs + wall_print_cost * wall_prints + installation_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_curtain_cost_calculation_l1285_128525


namespace NUMINAMATH_CALUDE_unique_prime_pair_squares_l1285_128544

theorem unique_prime_pair_squares : ∃! (p q : ℕ), 
  Prime p ∧ Prime q ∧ 
  ∃ (a b : ℕ), (p - q = a^2) ∧ (p*q - q = b^2) := by
sorry

end NUMINAMATH_CALUDE_unique_prime_pair_squares_l1285_128544


namespace NUMINAMATH_CALUDE_mari_buttons_l1285_128510

theorem mari_buttons (sue_buttons : ℕ) (kendra_buttons : ℕ) (mari_buttons : ℕ) 
  (h1 : sue_buttons = 6)
  (h2 : sue_buttons = kendra_buttons / 2)
  (h3 : mari_buttons = 5 * kendra_buttons + 4) :
  mari_buttons = 64 := by
  sorry

end NUMINAMATH_CALUDE_mari_buttons_l1285_128510


namespace NUMINAMATH_CALUDE_max_ratio_OB_OA_l1285_128529

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y = 1
def C₂ (x y φ : ℝ) : Prop := x = 2 + 2 * Real.cos φ ∧ y = 2 * Real.sin φ ∧ 0 ≤ φ ∧ φ < 2 * Real.pi

-- Define the ray l
def l (ρ θ α : ℝ) : Prop := θ = α ∧ ρ ≥ 0

-- Define points A and B
def A (ρ θ : ℝ) : Prop := C₁ (ρ * Real.cos θ) (ρ * Real.sin θ) ∧ l ρ θ θ
def B (ρ θ : ℝ) : Prop := ∃ φ, C₂ (ρ * Real.cos θ) (ρ * Real.sin θ) φ ∧ l ρ θ θ

-- State the theorem
theorem max_ratio_OB_OA :
  ∃ (max : ℝ), max = 2 + 2 * Real.sqrt 2 ∧
  ∀ α : ℝ, 0 ≤ α ∧ α ≤ Real.pi / 2 →
    ∀ ρA ρB θA θB : ℝ,
      A ρA θA → B ρB θB →
      ρB / ρA ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_ratio_OB_OA_l1285_128529


namespace NUMINAMATH_CALUDE_fraction_equality_l1285_128562

theorem fraction_equality (w x y : ℝ) (hw_x : w / x = 1 / 3) (hw_y : w / y = 3 / 4) 
  (hx : x ≠ 0) (hy : y ≠ 0) : (x + y) / y = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1285_128562


namespace NUMINAMATH_CALUDE_remainder_mod_11_l1285_128528

theorem remainder_mod_11 : (8735+100) + (8736+100) + (8737+100) + (8738+100) * 2 ≡ 10 [MOD 11] := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_11_l1285_128528


namespace NUMINAMATH_CALUDE_blue_ball_count_l1285_128511

/-- Given a bag of glass balls with yellow and blue colors -/
structure GlassBallBag where
  total : ℕ
  yellowProb : ℝ

/-- Theorem: In a bag of 80 glass balls where the probability of picking a yellow ball is 0.25,
    the number of blue balls is 60 -/
theorem blue_ball_count (bag : GlassBallBag)
    (h_total : bag.total = 80)
    (h_yellow_prob : bag.yellowProb = 0.25) :
    (bag.total : ℝ) * (1 - bag.yellowProb) = 60 := by
  sorry


end NUMINAMATH_CALUDE_blue_ball_count_l1285_128511


namespace NUMINAMATH_CALUDE_brenda_skittles_count_l1285_128592

def final_skittles (initial bought given_away : ℕ) : ℕ :=
  initial + bought - given_away

theorem brenda_skittles_count : final_skittles 7 8 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_brenda_skittles_count_l1285_128592


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l1285_128506

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all x. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2

/-- If 4x^2 + 2kx + 25 is a perfect square trinomial, then k = ±10. -/
theorem perfect_square_trinomial_condition (k : ℝ) :
  is_perfect_square_trinomial 4 (2*k) 25 → k = 10 ∨ k = -10 := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l1285_128506


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l1285_128564

/-- An isosceles right triangle with given area and hypotenuse length -/
structure IsoscelesRightTriangle where
  -- The length of a leg
  leg : ℝ
  -- The area of the triangle
  area : ℝ
  -- The condition that the area is equal to half the square of the leg
  area_eq : area = leg^2 / 2

/-- The theorem stating that an isosceles right triangle with area 9 has hypotenuse length 6 -/
theorem isosceles_right_triangle_hypotenuse (t : IsoscelesRightTriangle) 
  (h_area : t.area = 9) : 
  t.leg * Real.sqrt 2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l1285_128564


namespace NUMINAMATH_CALUDE_x_values_l1285_128563

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 12) (h2 : y + 1 / x = 7 / 18) :
  x = 6 + Real.sqrt 5 ∨ x = 6 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_x_values_l1285_128563


namespace NUMINAMATH_CALUDE_problem_1_l1285_128500

theorem problem_1 (x y : ℝ) (h : x^2 + y^2 = 1) :
  x^6 + 3*x^2*y^2 + y^6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1285_128500


namespace NUMINAMATH_CALUDE_joan_total_games_l1285_128539

/-- The total number of football games Joan attended over two years -/
def total_games (this_year last_year : ℕ) : ℕ :=
  this_year + last_year

/-- Theorem: Joan attended 9 football games in total over two years -/
theorem joan_total_games : total_games 4 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_joan_total_games_l1285_128539


namespace NUMINAMATH_CALUDE_dans_eggs_l1285_128586

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Dan bought -/
def dans_dozens : ℕ := 9

/-- Theorem: Dan bought 108 eggs -/
theorem dans_eggs : dans_dozens * eggs_per_dozen = 108 := by
  sorry

end NUMINAMATH_CALUDE_dans_eggs_l1285_128586


namespace NUMINAMATH_CALUDE_four_digit_multiples_of_seven_l1285_128594

theorem four_digit_multiples_of_seven : 
  (Finset.filter (fun n : ℕ => n % 7 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999) (Finset.range 10000)).card = 1286 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_multiples_of_seven_l1285_128594


namespace NUMINAMATH_CALUDE_luke_coin_count_l1285_128541

theorem luke_coin_count : 
  ∀ (quarter_piles dime_piles coins_per_pile : ℕ),
    quarter_piles = 5 →
    dime_piles = 5 →
    coins_per_pile = 3 →
    quarter_piles * coins_per_pile + dime_piles * coins_per_pile = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_luke_coin_count_l1285_128541


namespace NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l1285_128503

/-- Given a triangle with perimeter 48 and inradius 2.5, prove its area is 60 -/
theorem triangle_area_from_perimeter_and_inradius :
  ∀ (T : Set ℝ) (perimeter inradius area : ℝ),
  (perimeter = 48) →
  (inradius = 2.5) →
  (area = inradius * (perimeter / 2)) →
  area = 60 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l1285_128503


namespace NUMINAMATH_CALUDE_initial_pigs_count_l1285_128537

theorem initial_pigs_count (initial_cows initial_goats added_cows added_pigs added_goats total_after : ℕ) :
  initial_cows = 2 →
  initial_goats = 6 →
  added_cows = 3 →
  added_pigs = 5 →
  added_goats = 2 →
  total_after = 21 →
  ∃ initial_pigs : ℕ, 
    initial_cows + initial_pigs + initial_goats + added_cows + added_pigs + added_goats = total_after ∧
    initial_pigs = 3 :=
by sorry

end NUMINAMATH_CALUDE_initial_pigs_count_l1285_128537


namespace NUMINAMATH_CALUDE_elf_can_equalize_l1285_128593

/-- Represents the amount of milk in each cup -/
def MilkDistribution := Fin 30 → ℚ

/-- An operation that averages the milk in two cups -/
def average_cups (d : MilkDistribution) (i j : Fin 30) : MilkDistribution :=
  fun k => if k = i ∨ k = j then (d i + d j) / 2 else d k

/-- A sequence of cup-averaging operations -/
def OperationSequence := List (Fin 30 × Fin 30)

/-- Apply a sequence of operations to a milk distribution -/
def apply_operations (d : MilkDistribution) (ops : OperationSequence) : MilkDistribution :=
  ops.foldl (fun acc (i, j) => average_cups acc i j) d

/-- Check if all cups have the same amount of milk -/
def is_equalized (d : MilkDistribution) : Prop :=
  ∀ i j : Fin 30, d i = d j

/-- The main theorem: there always exists a finite sequence of operations to equalize any initial distribution -/
theorem elf_can_equalize (d : MilkDistribution) :
  ∃ (ops : OperationSequence), is_equalized (apply_operations d ops) := by
  sorry

end NUMINAMATH_CALUDE_elf_can_equalize_l1285_128593


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l1285_128572

/-- Represents a tetrahedron ABCD with given edge lengths -/
structure Tetrahedron where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  BD : ℝ
  AD : ℝ
  CD : ℝ

/-- Calculate the volume of a tetrahedron given its edge lengths -/
def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- The theorem stating that the volume of the specific tetrahedron is 24/5 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    AB := 5,
    AC := 3,
    BC := 4,
    BD := 4,
    AD := 3,
    CD := 12/5 * Real.sqrt 2
  }
  tetrahedronVolume t = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l1285_128572


namespace NUMINAMATH_CALUDE_marston_county_population_l1285_128583

theorem marston_county_population (num_cities : ℕ) (lower_bound upper_bound : ℝ) :
  num_cities = 25 →
  lower_bound = 4800 →
  upper_bound = 5200 →
  (num_cities : ℝ) * ((lower_bound + upper_bound) / 2) = 125000 := by
  sorry

end NUMINAMATH_CALUDE_marston_county_population_l1285_128583


namespace NUMINAMATH_CALUDE_grandmas_farm_l1285_128560

theorem grandmas_farm (chickens ducks : ℕ) : 
  chickens = 4 * ducks ∧ chickens = ducks + 600 → chickens = 800 ∧ ducks = 200 := by
  sorry

end NUMINAMATH_CALUDE_grandmas_farm_l1285_128560


namespace NUMINAMATH_CALUDE_lake_half_covered_l1285_128595

/-- Represents the number of lotuses in the lake on a given day -/
def lotuses (day : ℕ) : ℝ := 2^day

/-- The day when the lake is completely covered -/
def full_coverage_day : ℕ := 30

theorem lake_half_covered :
  lotuses (full_coverage_day - 1) = (1/2) * lotuses full_coverage_day :=
by sorry

end NUMINAMATH_CALUDE_lake_half_covered_l1285_128595


namespace NUMINAMATH_CALUDE_heather_aprons_tomorrow_l1285_128553

/-- The number of aprons Heather should sew tomorrow -/
def aprons_tomorrow (total : ℕ) (initial : ℕ) (today_multiplier : ℕ) : ℕ :=
  (total - (initial + today_multiplier * initial)) / 2

/-- Theorem: Given the conditions, Heather should sew 49 aprons tomorrow -/
theorem heather_aprons_tomorrow :
  aprons_tomorrow 150 13 3 = 49 := by
  sorry

end NUMINAMATH_CALUDE_heather_aprons_tomorrow_l1285_128553


namespace NUMINAMATH_CALUDE_greatest_gcd_with_linear_combination_l1285_128584

theorem greatest_gcd_with_linear_combination (m n : ℕ) : 
  Nat.gcd m n = 1 → 
  (∃ (a b : ℕ), Nat.gcd (m + 2000 * n) (n + 2000 * m) = a ∧ 
                a ≤ b ∧ 
                ∀ (c : ℕ), Nat.gcd (m + 2000 * n) (n + 2000 * m) ≤ c → c ≤ b) ∧
  3999999 = Nat.gcd (m + 2000 * n) (n + 2000 * m) := by
  sorry

end NUMINAMATH_CALUDE_greatest_gcd_with_linear_combination_l1285_128584


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1285_128566

/-- Given a geometric sequence {a_n} with a_1 = 1/8 and a_4 = -1, prove that the common ratio q is -2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a1 : a 1 = 1/8) 
  (h_a4 : a 4 = -1) 
  (q : ℚ) : 
  q = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1285_128566


namespace NUMINAMATH_CALUDE_cube_sum_gt_product_sum_l1285_128502

theorem cube_sum_gt_product_sum {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a^3 + b^3 > a^2*b + a*b^2 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_gt_product_sum_l1285_128502


namespace NUMINAMATH_CALUDE_existence_of_n_consecutive_representable_l1285_128568

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Part 1: Existence of n such that n + S(n) = 1980
theorem existence_of_n : ∃ n : ℕ, n + S n = 1980 := by sorry

-- Part 2: For any m, either m or m+1 can be expressed as n + S(n)
theorem consecutive_representable (m : ℕ) : 
  (∃ n : ℕ, n + S n = m) ∨ (∃ n : ℕ, n + S n = m + 1) := by sorry

end NUMINAMATH_CALUDE_existence_of_n_consecutive_representable_l1285_128568


namespace NUMINAMATH_CALUDE_sequence_inequality_l1285_128534

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, a n ≥ 0) 
  (h_ineq : ∀ m n, a (m + n) ≤ a m + a n) (m n : ℕ) (h_ge : n ≥ m) :
  a n ≤ m * a 1 + (n / m - 1) * a m :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1285_128534


namespace NUMINAMATH_CALUDE_sum_of_median_scores_l1285_128565

def median_score_A : ℕ := 28
def median_score_B : ℕ := 36

theorem sum_of_median_scores :
  median_score_A + median_score_B = 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_median_scores_l1285_128565


namespace NUMINAMATH_CALUDE_triangle_existence_and_uniqueness_l1285_128554

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the given points
variable (D E F : Point)

-- Define the conditions
def is_midpoint (M : Point) (A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def is_trisection_point (E B C : Point) : Prop :=
  E.x = B.x + (C.x - B.x) / 3 ∧ E.y = B.y + (C.y - B.y) / 3

def is_quarter_point (F C A : Point) : Prop :=
  F.x = C.x + 3 * (A.x - C.x) / 4 ∧ F.y = C.y + 3 * (A.y - C.y) / 4

-- State the theorem
theorem triangle_existence_and_uniqueness :
  ∃! (ABC : Triangle),
    is_midpoint D ABC.A ABC.B ∧
    is_trisection_point E ABC.B ABC.C ∧
    is_quarter_point F ABC.C ABC.A :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_and_uniqueness_l1285_128554


namespace NUMINAMATH_CALUDE_natural_number_divisibility_l1285_128559

theorem natural_number_divisibility (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by sorry

end NUMINAMATH_CALUDE_natural_number_divisibility_l1285_128559


namespace NUMINAMATH_CALUDE_max_sum_hexagonal_prism_with_pyramid_l1285_128596

/-- Represents a three-dimensional geometric shape -/
structure Shape3D where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- A hexagonal prism -/
def hexagonal_prism : Shape3D :=
  { faces := 8, vertices := 12, edges := 18 }

/-- Adds a pyramid to one face of a given shape -/
def add_pyramid (s : Shape3D) : Shape3D :=
  { faces := s.faces + 5,  -- Loses 1 face, gains 6
    vertices := s.vertices + 1,
    edges := s.edges + 6 }

/-- Calculates the sum of faces, vertices, and edges -/
def shape_sum (s : Shape3D) : ℕ :=
  s.faces + s.vertices + s.edges

/-- Theorem: The maximum sum of faces, vertices, and edges after adding a pyramid to a hexagonal prism is 44 -/
theorem max_sum_hexagonal_prism_with_pyramid :
  shape_sum (add_pyramid hexagonal_prism) = 44 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_hexagonal_prism_with_pyramid_l1285_128596


namespace NUMINAMATH_CALUDE_amanda_candy_bars_l1285_128580

theorem amanda_candy_bars :
  let initial_candy_bars : ℕ := 7
  let first_giveaway : ℕ := 3
  let bought_candy_bars : ℕ := 30
  let second_giveaway_multiplier : ℕ := 4
  
  let remaining_after_first := initial_candy_bars - first_giveaway
  let second_giveaway := first_giveaway * second_giveaway_multiplier
  let remaining_after_second := bought_candy_bars - second_giveaway
  let total_kept := remaining_after_first + remaining_after_second

  total_kept = 22 := by
  sorry

end NUMINAMATH_CALUDE_amanda_candy_bars_l1285_128580


namespace NUMINAMATH_CALUDE_die_events_l1285_128561

-- Define the sample space and events
def Ω : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1}
def B : Set Nat := {2, 4, 6}
def C : Set Nat := {1, 2}
def D : Set Nat := {3, 4, 5, 6}
def E : Set Nat := {3, 6}

-- Theorem to prove the relationships and set operations
theorem die_events :
  (A ⊆ C) ∧
  (C ∪ D = Ω) ∧
  (E ⊆ D) ∧
  (Dᶜ = {1, 2}) ∧
  (Aᶜ ∩ C = {2}) ∧
  (Bᶜ ∪ C = {1, 2, 3}) ∧
  (Dᶜ ∪ Eᶜ = {1, 2, 4, 5}) :=
by sorry

end NUMINAMATH_CALUDE_die_events_l1285_128561


namespace NUMINAMATH_CALUDE_max_removed_squares_elegantly_destroyed_l1285_128501

/-- Represents a chessboard --/
structure Chessboard :=
  (size : Nat)
  (removed : Finset (Nat × Nat))

/-- Represents a domino --/
inductive Domino
  | Horizontal : Nat → Nat → Domino
  | Vertical : Nat → Nat → Domino

/-- Checks if a domino can be placed on the board --/
def canPlaceDomino (board : Chessboard) (d : Domino) : Prop :=
  match d with
  | Domino.Horizontal x y => 
      x < board.size ∧ y < board.size - 1 ∧ 
      (x, y) ∉ board.removed ∧ (x, y + 1) ∉ board.removed
  | Domino.Vertical x y => 
      x < board.size - 1 ∧ y < board.size ∧ 
      (x, y) ∉ board.removed ∧ (x + 1, y) ∉ board.removed

/-- Defines an "elegantly destroyed" board --/
def isElegantlyDestroyed (board : Chessboard) : Prop :=
  (∀ d : Domino, ¬canPlaceDomino board d) ∧
  (∀ s : Nat × Nat, s ∈ board.removed →
    ∃ d : Domino, canPlaceDomino { size := board.size, removed := board.removed.erase s } d)

/-- The main theorem --/
theorem max_removed_squares_elegantly_destroyed :
  ∃ (board : Chessboard),
    board.size = 8 ∧
    isElegantlyDestroyed board ∧
    board.removed.card = 48 ∧
    (∀ (board' : Chessboard), board'.size = 8 →
      isElegantlyDestroyed board' →
      board'.removed.card ≤ 48) :=
  sorry

end NUMINAMATH_CALUDE_max_removed_squares_elegantly_destroyed_l1285_128501


namespace NUMINAMATH_CALUDE_sam_speed_l1285_128532

/-- Given the biking speeds of Eugene, Clara, and Sam, prove Sam's speed --/
theorem sam_speed (eugene_speed : ℚ) (clara_ratio : ℚ) (sam_ratio : ℚ) :
  eugene_speed = 5 →
  clara_ratio = 3 / 4 →
  sam_ratio = 4 / 3 →
  sam_ratio * (clara_ratio * eugene_speed) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sam_speed_l1285_128532


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l1285_128522

theorem min_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) : 
  (1 / ((1 - x) * (1 - y) * (1 - z))) + (1 / ((1 + x) * (1 + y) * (1 + z))) + x * y * z ≥ 2 :=
by sorry

theorem min_value_achieved (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) : 
  (1 / ((1 - x) * (1 - y) * (1 - z))) + (1 / ((1 + x) * (1 + y) * (1 + z))) + x * y * z = 2 ↔ 
  x = 0 ∧ y = 0 ∧ z = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l1285_128522


namespace NUMINAMATH_CALUDE_same_color_probability_l1285_128587

def total_balls : ℕ := 20
def blue_balls : ℕ := 8
def green_balls : ℕ := 5
def red_balls : ℕ := 7

theorem same_color_probability :
  let prob_blue := (blue_balls / total_balls) ^ 2
  let prob_green := (green_balls / total_balls) ^ 2
  let prob_red := (red_balls / total_balls) ^ 2
  prob_blue + prob_green + prob_red = 117 / 200 := by
sorry

end NUMINAMATH_CALUDE_same_color_probability_l1285_128587


namespace NUMINAMATH_CALUDE_one_fifth_equals_point_two_l1285_128567

theorem one_fifth_equals_point_two : (1 : ℚ) / 5 = 0.200000 := by sorry

end NUMINAMATH_CALUDE_one_fifth_equals_point_two_l1285_128567


namespace NUMINAMATH_CALUDE_pencils_leftover_l1285_128526

theorem pencils_leftover : 76394821 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencils_leftover_l1285_128526


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_3_l1285_128519

theorem no_solution_implies_a_leq_3 :
  (∀ x : ℝ, ¬(x > 3 ∧ x < a)) → a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_3_l1285_128519


namespace NUMINAMATH_CALUDE_christopher_sugar_substitute_cost_l1285_128512

/-- Represents the cost calculation for Christopher's sugar substitute usage --/
theorem christopher_sugar_substitute_cost :
  let packets_per_coffee : ℕ := 1
  let coffees_per_day : ℕ := 2
  let packets_per_box : ℕ := 30
  let cost_per_box : ℚ := 4
  let days : ℕ := 90

  let daily_usage : ℕ := packets_per_coffee * coffees_per_day
  let total_packets : ℕ := daily_usage * days
  let boxes_needed : ℕ := (total_packets + packets_per_box - 1) / packets_per_box
  let total_cost : ℚ := cost_per_box * boxes_needed

  total_cost = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_christopher_sugar_substitute_cost_l1285_128512


namespace NUMINAMATH_CALUDE_card_distribution_l1285_128571

theorem card_distribution (n : ℕ) : 
  (Finset.sum (Finset.range (n - 1)) (λ k => Nat.choose n (k + 1))) = 2 * (2^(n - 1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_card_distribution_l1285_128571


namespace NUMINAMATH_CALUDE_trajectory_of_m_l1285_128579

/-- The trajectory of point M given conditions on triangle MAB -/
theorem trajectory_of_m (x y : ℝ) (hx : x ≠ 3 ∧ x ≠ -3) (hy : y ≠ 0) :
  (y / (x + 3)) * (y / (x - 3)) = 4 →
  x^2 / 9 - y^2 / 36 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_m_l1285_128579


namespace NUMINAMATH_CALUDE_expression_evaluation_l1285_128582

theorem expression_evaluation :
  let x : ℝ := -5
  let y : ℝ := 8
  let z : ℝ := 3
  let w : ℝ := 2
  Real.sqrt (2 * z * (w - y)^2 - x^3 * y) + Real.sin (Real.pi * z) * x * w^2 - Real.tan (Real.pi * x^2) * z^3 = Real.sqrt 1216 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1285_128582


namespace NUMINAMATH_CALUDE_expression_simplification_l1285_128556

theorem expression_simplification (x : ℝ) 
  (h1 : x ≠ 2) (h2 : x ≠ 4) (h3 : x ≠ 3) (h4 : x ≠ 5) :
  (x^2 - 2*x + 1) / (x^2 - 6*x + 8) / ((x^2 - 4*x + 3) / (x^2 - 8*x + 15)) = (x - 5) / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1285_128556


namespace NUMINAMATH_CALUDE_parallel_lines_intersection_problem_solution_l1285_128517

/-- Given two sets of parallel lines intersecting each other, 
    calculate the number of lines in the second set based on 
    the number of parallelograms formed -/
theorem parallel_lines_intersection (first_set : ℕ) (parallelograms : ℕ) 
  (h1 : first_set = 5) 
  (h2 : parallelograms = 280) : 
  ∃ (second_set : ℕ), second_set * (first_set - 1) = parallelograms := by
  sorry

/-- The specific case for the given problem -/
theorem problem_solution : 
  ∃ (second_set : ℕ), second_set * 4 = 280 ∧ second_set = 71 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_intersection_problem_solution_l1285_128517


namespace NUMINAMATH_CALUDE_perimeter_of_triangle_cos_A_minus_C_l1285_128545

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = 1 ∧ b = 2 ∧ Real.cos C = 1/4

-- Theorem for the perimeter
theorem perimeter_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : a + b + c = 5 := by
  sorry

-- Theorem for cos(A-C)
theorem cos_A_minus_C (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : Real.cos (A - C) = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_triangle_cos_A_minus_C_l1285_128545


namespace NUMINAMATH_CALUDE_triangle_side_equation_l1285_128530

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the altitude equations
def altitude1 (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0
def altitude2 (x y : ℝ) : Prop := x + y = 0

-- Define the theorem
theorem triangle_side_equation (ABC : Triangle) 
  (h1 : ABC.A = (1, 2))
  (h2 : altitude1 (ABC.B.1) (ABC.B.2) ∨ altitude1 (ABC.C.1) (ABC.C.2))
  (h3 : altitude2 (ABC.B.1) (ABC.B.2) ∨ altitude2 (ABC.C.1) (ABC.C.2)) :
  ∃ (a b c : ℝ), a * ABC.B.1 + b * ABC.B.2 + c = 0 ∧
                 a * ABC.C.1 + b * ABC.C.2 + c = 0 ∧
                 (a, b, c) = (2, 3, 7) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_equation_l1285_128530


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l1285_128508

theorem book_arrangement_theorem :
  let math_books : ℕ := 4
  let english_books : ℕ := 4
  let group_arrangements : ℕ := 2  -- math books and English books as two groups
  let total_arrangements : ℕ := group_arrangements.factorial * math_books.factorial * english_books.factorial
  total_arrangements = 1152 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l1285_128508


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_eq_neg_one_l1285_128535

/-- Two vectors in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Definition of parallel vectors in R² -/
def parallel (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v.x = k * w.x ∧ v.y = k * w.y

/-- The main theorem -/
theorem parallel_vectors_imply_x_eq_neg_one :
  ∀ (x : ℝ),
  let a : Vector2D := ⟨x, 1⟩
  let b : Vector2D := ⟨1, -1⟩
  parallel a b → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_eq_neg_one_l1285_128535


namespace NUMINAMATH_CALUDE_square_perimeter_no_conditional_l1285_128590

-- Define the problem types
inductive Problem
| OppositeNumber
| SquarePerimeter
| MaximumOfThree
| BinaryToDecimal

-- Define a predicate for problems that don't require conditional statements
def NoConditionalRequired (p : Problem) : Prop :=
  match p with
  | Problem.SquarePerimeter => True
  | _ => False

-- Theorem statement
theorem square_perimeter_no_conditional :
  NoConditionalRequired Problem.SquarePerimeter ∧
  ¬NoConditionalRequired Problem.OppositeNumber ∧
  ¬NoConditionalRequired Problem.MaximumOfThree ∧
  ¬NoConditionalRequired Problem.BinaryToDecimal :=
sorry

end NUMINAMATH_CALUDE_square_perimeter_no_conditional_l1285_128590


namespace NUMINAMATH_CALUDE_erased_number_proof_l1285_128520

theorem erased_number_proof (n : ℕ) (x : ℕ) :
  n > 0 →
  x > 0 →
  x ≤ n →
  (n * (n + 1) / 2 - x) / (n - 1) = 182 / 5 →
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l1285_128520


namespace NUMINAMATH_CALUDE_power_of_two_between_powers_of_ten_l1285_128574

theorem power_of_two_between_powers_of_ten (t : ℕ+) : 
  (10 ^ (t.val - 1 : ℕ) < 2 ^ 64) ∧ (2 ^ 64 < 10 ^ t.val) → t = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_between_powers_of_ten_l1285_128574


namespace NUMINAMATH_CALUDE_warrior_truth_count_l1285_128513

/-- Represents the types of weapons a warrior can have as their favorite. -/
inductive Weapon
| sword
| spear
| axe
| bow

/-- Represents a warrior's truthfulness. -/
inductive Truthfulness
| truthful
| liar

/-- Represents the problem setup. -/
structure WarriorProblem where
  totalWarriors : Nat
  swordYes : Nat
  spearYes : Nat
  axeYes : Nat
  bowYes : Nat

/-- The main theorem to prove. -/
theorem warrior_truth_count (problem : WarriorProblem)
  (h_total : problem.totalWarriors = 33)
  (h_sword : problem.swordYes = 13)
  (h_spear : problem.spearYes = 15)
  (h_axe : problem.axeYes = 20)
  (h_bow : problem.bowYes = 27)
  : { truthfulCount : Nat // 
      truthfulCount = 12 ∧
      truthfulCount + (problem.totalWarriors - truthfulCount) * 3 = 
        problem.swordYes + problem.spearYes + problem.axeYes + problem.bowYes } :=
  sorry


end NUMINAMATH_CALUDE_warrior_truth_count_l1285_128513


namespace NUMINAMATH_CALUDE_line_conditions_l1285_128573

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ

-- Define the lines from the problem
def line1 : Line := λ x y => x + y - 1
def line2 : Line := λ x y => x + y - 2
def line3 : Line := λ x y => x - 3*y + 3
def line4 : Line := λ x y => 3*x + y + 1

-- Define the points from the problem
def point1 : Point := (-1, 2)
def point2 : Point := (0, 1)

-- Define what it means for a line to pass through a point
def passes_through (l : Line) (p : Point) : Prop :=
  l p.1 p.2 = 0

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l1 x y = k * l2 x y

-- Define what it means for two lines to be perpendicular
def perpendicular (l1 l2 : Line) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l1 x y = k * l2 y (-x)

-- State the theorem
theorem line_conditions : 
  (passes_through line1 point1 ∧ parallel line1 line2) ∧
  (passes_through line3 point2 ∧ perpendicular line3 line4) := by
  sorry

end NUMINAMATH_CALUDE_line_conditions_l1285_128573


namespace NUMINAMATH_CALUDE_evaluate_expression_l1285_128585

theorem evaluate_expression : 500 * (500^500) * 500 = 500^502 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1285_128585


namespace NUMINAMATH_CALUDE_fraction_reduction_l1285_128552

theorem fraction_reduction (a x : ℝ) :
  (Real.sqrt (a^2 + x^2) - (x^2 - a^2) / Real.sqrt (a^2 + x^2)) / (a^2 + x^2) = 
  2 * a^2 / (a^2 + x^2)^(3/2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_reduction_l1285_128552


namespace NUMINAMATH_CALUDE_union_when_m_neg_one_subset_iff_m_range_disjoint_iff_m_range_l1285_128531

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Theorem 1
theorem union_when_m_neg_one :
  A ∪ B (-1) = {x | -2 < x ∧ x < 3} := by sorry

-- Theorem 2
theorem subset_iff_m_range :
  ∀ m, A ⊆ B m ↔ m ≤ -2 := by sorry

-- Theorem 3
theorem disjoint_iff_m_range :
  ∀ m, A ∩ B m = ∅ ↔ 0 ≤ m := by sorry

end NUMINAMATH_CALUDE_union_when_m_neg_one_subset_iff_m_range_disjoint_iff_m_range_l1285_128531


namespace NUMINAMATH_CALUDE_rope_remaining_lengths_l1285_128569

/-- Calculates the remaining lengths of two ropes after giving away portions. -/
theorem rope_remaining_lengths (x y : ℝ) (p q : ℝ) : 
  p = 0.40 * x ∧ q = 0.5625 * y := by
  sorry

#check rope_remaining_lengths

end NUMINAMATH_CALUDE_rope_remaining_lengths_l1285_128569


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1285_128514

theorem sqrt_equation_solution :
  ∀ a b : ℕ+,
    a < b →
    (Real.sqrt (1 + Real.sqrt (25 + 20 * Real.sqrt 2)) = Real.sqrt a + Real.sqrt b) →
    (a = 2 ∧ b = 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1285_128514


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1285_128536

theorem inequality_system_solution (x : ℝ) :
  (1/3 * x - 1 ≤ 1/2 * x + 1) →
  (3 * x - (x - 2) ≥ 6) →
  (x + 1 > (4 * x - 1) / 3) →
  (2 ≤ x ∧ x < 4) := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1285_128536


namespace NUMINAMATH_CALUDE_min_airlines_needed_l1285_128551

/-- Represents the number of towns -/
def num_towns : ℕ := 21

/-- Represents the size of the group of towns served by each airline -/
def group_size : ℕ := 5

/-- Calculates the total number of pairs of towns -/
def total_pairs : ℕ := num_towns.choose 2

/-- Calculates the number of pairs served by each airline -/
def pairs_per_airline : ℕ := group_size.choose 2

/-- Theorem stating the minimum number of airlines needed -/
theorem min_airlines_needed : 
  ∃ (n : ℕ), n * pairs_per_airline ≥ total_pairs ∧ 
  ∀ (m : ℕ), m * pairs_per_airline ≥ total_pairs → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_min_airlines_needed_l1285_128551


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1285_128577

/-- Circle O₁ with equation x² + y² = 1 -/
def circle_O₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Circle O₂ with equation x² + y² - 6x + 8y + 9 = 0 -/
def circle_O₂ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 + 8*p.2 + 9 = 0}

/-- The center of circle O₁ -/
def center_O₁ : ℝ × ℝ := (0, 0)

/-- The radius of circle O₁ -/
def radius_O₁ : ℝ := 1

/-- The center of circle O₂ -/
def center_O₂ : ℝ × ℝ := (3, -4)

/-- The radius of circle O₂ -/
def radius_O₂ : ℝ := 4

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c₁ c₂ : Set (ℝ × ℝ)) (center₁ center₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  ((center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2).sqrt = r₁ + r₂

theorem circles_externally_tangent :
  externally_tangent circle_O₁ circle_O₂ center_O₁ center_O₂ radius_O₁ radius_O₂ := by
  sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l1285_128577


namespace NUMINAMATH_CALUDE_pentagon_coverage_l1285_128515

-- Define a pentagon as a set of 5 points in 2D space
def Pentagon : Type := Fin 5 → ℝ × ℝ

-- Define a function to check if a pentagon is convex
def isConvex (p : Pentagon) : Prop := sorry

-- Define a function to check if all interior angles of a pentagon are obtuse
def allAnglesObtuse (p : Pentagon) : Prop := sorry

-- Define a function to check if a point is inside or on a circle
def isInsideOrOnCircle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop := sorry

-- Define a function to check if a circle covers a point of the pentagon
def circleCoversPoint (p : Pentagon) (diagonal : Fin 5 × Fin 5) (point : Fin 5) : Prop := sorry

-- Main theorem
theorem pentagon_coverage (p : Pentagon) 
  (h_convex : isConvex p) 
  (h_obtuse : allAnglesObtuse p) : 
  ∃ (d1 d2 : Fin 5 × Fin 5), ∀ (point : Fin 5), 
    circleCoversPoint p d1 point ∨ circleCoversPoint p d2 point := by
  sorry

end NUMINAMATH_CALUDE_pentagon_coverage_l1285_128515


namespace NUMINAMATH_CALUDE_last_digit_periodic_l1285_128524

theorem last_digit_periodic (n : ℕ) : n^n % 10 = (n + 20)^(n + 20) % 10 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_periodic_l1285_128524


namespace NUMINAMATH_CALUDE_cosine_inequality_l1285_128589

theorem cosine_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1/2) : 
  Real.cos (1 + a) < Real.cos (1 - a) := by
sorry

end NUMINAMATH_CALUDE_cosine_inequality_l1285_128589


namespace NUMINAMATH_CALUDE_craig_and_mother_age_difference_l1285_128549

/-- Craig and his mother's ages problem -/
theorem craig_and_mother_age_difference :
  ∀ (craig_age mother_age : ℕ),
    craig_age + mother_age = 56 →
    craig_age = 16 →
    mother_age - craig_age = 24 := by
  sorry

end NUMINAMATH_CALUDE_craig_and_mother_age_difference_l1285_128549


namespace NUMINAMATH_CALUDE_safe_count_theorem_l1285_128504

def is_p_safe (n p : ℕ) : Prop :=
  n % p > 2 ∧ n % p < p - 2

def count_safe (max : ℕ) : ℕ :=
  (max / (5 * 7 * 17)) * 48

theorem safe_count_theorem :
  count_safe 20000 = 1584 ∧
  ∀ n : ℕ, n ≤ 20000 →
    (is_p_safe n 5 ∧ is_p_safe n 7 ∧ is_p_safe n 17) ↔
    ∃ k : ℕ, k < 48 ∧ n ≡ k [MOD 595] :=
by sorry

end NUMINAMATH_CALUDE_safe_count_theorem_l1285_128504


namespace NUMINAMATH_CALUDE_crackers_box_sleeves_l1285_128505

/-- The number of crackers Chad uses per sandwich -/
def crackers_per_sandwich : ℕ := 2

/-- The number of sandwiches Chad eats per night -/
def sandwiches_per_night : ℕ := 5

/-- The number of crackers in each sleeve -/
def crackers_per_sleeve : ℕ := 28

/-- The number of boxes of crackers -/
def num_boxes : ℕ := 5

/-- The number of nights the crackers last -/
def num_nights : ℕ := 56

/-- The number of sleeves in a box of crackers -/
def sleeves_per_box : ℕ := 4

theorem crackers_box_sleeves :
  sleeves_per_box = 4 :=
sorry

end NUMINAMATH_CALUDE_crackers_box_sleeves_l1285_128505


namespace NUMINAMATH_CALUDE_sum_of_four_squares_sum_of_four_squares_proof_l1285_128555

theorem sum_of_four_squares : ℕ → ℕ → ℕ → Prop :=
  fun triangle circle square =>
    triangle + circle + triangle + square = 27 ∧
    circle + triangle + circle + square = 25 ∧
    square + square + square + triangle = 39 →
    4 * square = 44

-- The proof would go here, but we're skipping it as per instructions
theorem sum_of_four_squares_proof (triangle circle square : ℕ) 
  (h : sum_of_four_squares triangle circle square) : 4 * square = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_squares_sum_of_four_squares_proof_l1285_128555


namespace NUMINAMATH_CALUDE_circles_common_chord_common_chord_length_l1285_128598

/-- Circle C₁ with equation x² + y² - 2x + 10y - 24 = 0 -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 10*y - 24 = 0

/-- Circle C₂ with equation x² + y² + 2x + 2y - 8 = 0 -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y - 8 = 0

/-- The line on which the common chord of C₁ and C₂ lies -/
def common_chord_line (x y : ℝ) : Prop :=
  x - 6*y + 6 = 0

theorem circles_common_chord (x y : ℝ) :
  (C₁ x y ∧ C₂ x y) → common_chord_line x y :=
sorry

theorem common_chord_length : 
  ∃ (a b : ℝ), C₁ a b ∧ C₂ a b ∧ 
  ∃ (c d : ℝ), C₁ c d ∧ C₂ c d ∧ 
  ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = 2 * 13^(1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_circles_common_chord_common_chord_length_l1285_128598


namespace NUMINAMATH_CALUDE_fraction_equality_l1285_128581

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 5) : 
  (5 * m * r - 2 * n * t) / (7 * n * t - 10 * m * r) = -31 / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1285_128581


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1285_128550

/-- A sequence a : ℕ → ℝ is geometric if there exists a non-zero real number r 
    such that for all n, a(n+1) = r * a(n) -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : IsGeometric a) 
  (h2 : a 3 * a 5 = 64) : a 4 = 8 ∨ a 4 = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1285_128550


namespace NUMINAMATH_CALUDE_circle_equation_through_origin_l1285_128570

/-- The equation of a circle with center (1, 1) passing through the origin (0, 0) is (x-1)^2 + (y-1)^2 = 2 -/
theorem circle_equation_through_origin (x y : ℝ) :
  let center : ℝ × ℝ := (1, 1)
  let origin : ℝ × ℝ := (0, 0)
  let on_circle (p : ℝ × ℝ) := (p.1 - center.1)^2 + (p.2 - center.2)^2 = (center.1 - origin.1)^2 + (center.2 - origin.2)^2
  on_circle (x, y) ↔ (x - 1)^2 + (y - 1)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_through_origin_l1285_128570


namespace NUMINAMATH_CALUDE_tiles_per_row_l1285_128546

-- Define the room area in square feet
def room_area : ℝ := 144

-- Define the tile size in inches
def tile_size : ℝ := 8

-- Define the number of inches in a foot
def inches_per_foot : ℝ := 12

-- Theorem to prove
theorem tiles_per_row : 
  ⌊(inches_per_foot * (room_area ^ (1/2 : ℝ))) / tile_size⌋ = 18 := by
  sorry

end NUMINAMATH_CALUDE_tiles_per_row_l1285_128546


namespace NUMINAMATH_CALUDE_number_at_2002_2003_l1285_128557

/-- Represents the number at position (row, col) in the arrangement -/
def number_at_position (row : ℕ) (col : ℕ) : ℕ :=
  (col - 1)^2 + 1 + (row - 1)

/-- The theorem to be proved -/
theorem number_at_2002_2003 :
  number_at_position 2002 2003 = 2002 * 2003 := by
  sorry

#check number_at_2002_2003

end NUMINAMATH_CALUDE_number_at_2002_2003_l1285_128557


namespace NUMINAMATH_CALUDE_novel_reading_time_l1285_128576

theorem novel_reading_time (total_pages : ℕ) (rate_alice rate_bob rate_chandra : ℚ) :
  total_pages = 760 ∧ 
  rate_alice = 1 / 20 ∧ 
  rate_bob = 1 / 45 ∧ 
  rate_chandra = 1 / 30 →
  ∃ t : ℚ, t = 7200 ∧ 
    t * rate_alice + t * rate_bob + t * rate_chandra = total_pages :=
by sorry

end NUMINAMATH_CALUDE_novel_reading_time_l1285_128576


namespace NUMINAMATH_CALUDE_decimal_addition_l1285_128516

theorem decimal_addition : (7.15 : ℝ) + 2.639 = 9.789 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_l1285_128516


namespace NUMINAMATH_CALUDE_range_of_m_l1285_128597

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2 - x - y = 0

-- Define the point A
def A (m : ℝ) : ℝ × ℝ := (m, m)

-- Define the condition that any line through A intersects C
def intersects_C (m : ℝ) : Prop :=
  ∀ (k b : ℝ), ∃ (x y : ℝ), C x y ∧ y = k * x + b ∧ m * k + b = m

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, intersects_C m ↔ 0 ≤ m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1285_128597


namespace NUMINAMATH_CALUDE_diophantine_equations_solutions_l1285_128507

-- Define the set of solutions for the first equation
def S₁ : Set (ℤ × ℤ) := {(x, y) | ∃ k : ℤ, x = 3 * k + 1 ∧ y = -2 * k + 1}

-- Define the set of solutions for the second equation
def S₂ : Set (ℤ × ℤ) := {(x, y) | ∃ k : ℤ, x = 5 * k ∧ y = 2 - 2 * k}

theorem diophantine_equations_solutions :
  (∀ (x y : ℤ), (2 * x + 3 * y = 5) ↔ (x, y) ∈ S₁) ∧
  (∀ (x y : ℤ), (2 * x + 5 * y = 10) ↔ (x, y) ∈ S₂) ∧
  (¬ ∃ (x y : ℤ), 3 * x + 9 * y = 2018) := by
  sorry

#check diophantine_equations_solutions

end NUMINAMATH_CALUDE_diophantine_equations_solutions_l1285_128507
