import Mathlib

namespace eraser_boxes_donated_l596_59690

theorem eraser_boxes_donated (erasers_per_box : ℕ) (price_per_eraser : ℚ) (total_money : ℚ) :
  erasers_per_box = 24 →
  price_per_eraser = 3/4 →
  total_money = 864 →
  (total_money / price_per_eraser) / erasers_per_box = 48 :=
by sorry

end eraser_boxes_donated_l596_59690


namespace valid_diagonals_150_sided_polygon_l596_59606

/-- The number of sides in the polygon -/
def n : ℕ := 150

/-- The total number of diagonals in an n-sided polygon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals to be excluded (those connecting vertices whose indices differ by a multiple of 4) -/
def excluded_diagonals (n : ℕ) : ℕ := n * (n / 4)

/-- The number of valid diagonals in the polygon -/
def valid_diagonals (n : ℕ) : ℕ := total_diagonals n - excluded_diagonals n

theorem valid_diagonals_150_sided_polygon :
  valid_diagonals n = 5400 := by
  sorry


end valid_diagonals_150_sided_polygon_l596_59606


namespace digit_product_sum_l596_59650

/-- A function that converts a pair of digits to a two-digit integer -/
def twoDigitInt (tens ones : Nat) : Nat :=
  10 * tens + ones

/-- A predicate that checks if a number is a positive digit (1-9) -/
def isPositiveDigit (n : Nat) : Prop :=
  0 < n ∧ n ≤ 9

theorem digit_product_sum (p q r : Nat) : 
  isPositiveDigit p ∧ isPositiveDigit q ∧ isPositiveDigit r →
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (twoDigitInt p q) * (twoDigitInt p r) = 221 →
  p + q + r = 11 →
  q = 7 := by
  sorry

end digit_product_sum_l596_59650


namespace sqrt_not_defined_for_negative_one_l596_59627

theorem sqrt_not_defined_for_negative_one :
  ¬ (∃ (y : ℝ), y^2 = -1) :=
sorry

end sqrt_not_defined_for_negative_one_l596_59627


namespace select_medical_team_eq_630_l596_59676

/-- The number of ways to select a medical team for earthquake relief. -/
def select_medical_team : ℕ :=
  let orthopedic : ℕ := 3
  let neurosurgeon : ℕ := 4
  let internist : ℕ := 5
  let team_size : ℕ := 5
  
  -- Combinations for each possible selection scenario
  let scenario1 := Nat.choose orthopedic 3 * Nat.choose neurosurgeon 1 * Nat.choose internist 1
  let scenario2 := Nat.choose orthopedic 1 * Nat.choose neurosurgeon 3 * Nat.choose internist 1
  let scenario3 := Nat.choose orthopedic 1 * Nat.choose neurosurgeon 1 * Nat.choose internist 3
  let scenario4 := Nat.choose orthopedic 2 * Nat.choose neurosurgeon 2 * Nat.choose internist 1
  let scenario5 := Nat.choose orthopedic 1 * Nat.choose neurosurgeon 2 * Nat.choose internist 2
  let scenario6 := Nat.choose orthopedic 2 * Nat.choose neurosurgeon 1 * Nat.choose internist 2

  -- Sum of all scenarios
  scenario1 + scenario2 + scenario3 + scenario4 + scenario5 + scenario6

/-- Theorem stating that the number of ways to select the medical team is 630. -/
theorem select_medical_team_eq_630 : select_medical_team = 630 := by
  sorry

end select_medical_team_eq_630_l596_59676


namespace rosa_initial_flowers_l596_59649

theorem rosa_initial_flowers (flowers_from_andre : ℕ) (total_flowers : ℕ) 
  (h1 : flowers_from_andre = 23)
  (h2 : total_flowers = 90) :
  total_flowers - flowers_from_andre = 67 := by
  sorry

end rosa_initial_flowers_l596_59649


namespace smaller_two_digit_factor_l596_59605

theorem smaller_two_digit_factor (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 4851 → 
  min a b = 49 := by
sorry

end smaller_two_digit_factor_l596_59605


namespace cube_root_simplification_l596_59697

theorem cube_root_simplification :
  (80^3 + 100^3 + 120^3 : ℝ)^(1/3) = 20 * 405^(1/3) :=
by sorry

end cube_root_simplification_l596_59697


namespace zaras_goats_l596_59680

theorem zaras_goats (cows sheep : ℕ) (groups : ℕ) (animals_per_group : ℕ) : 
  cows = 24 → 
  sheep = 7 → 
  groups = 3 → 
  animals_per_group = 48 → 
  groups * animals_per_group - cows - sheep = 113 :=
by sorry

end zaras_goats_l596_59680


namespace product_expansion_l596_59660

theorem product_expansion (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end product_expansion_l596_59660


namespace company_capital_expenditure_l596_59622

theorem company_capital_expenditure (C : ℝ) (C_pos : C > 0) :
  let raw_material := (1 / 4 : ℝ) * C
  let remaining_after_raw := C - raw_material
  let machinery := (1 / 10 : ℝ) * remaining_after_raw
  let capital_left := C - raw_material - machinery
  capital_left = (27 / 40 : ℝ) * C :=
by sorry

end company_capital_expenditure_l596_59622


namespace potato_rows_count_l596_59686

/-- Represents the farmer's crop situation -/
structure FarmCrops where
  corn_rows : ℕ
  potato_rows : ℕ
  corn_per_row : ℕ
  potatoes_per_row : ℕ
  intact_crops : ℕ

/-- Theorem stating the number of potato rows given the problem conditions -/
theorem potato_rows_count (farm : FarmCrops)
    (h_corn_rows : farm.corn_rows = 10)
    (h_corn_per_row : farm.corn_per_row = 9)
    (h_potatoes_per_row : farm.potatoes_per_row = 30)
    (h_intact_crops : farm.intact_crops = 120)
    (h_half_destroyed : farm.intact_crops = (farm.corn_rows * farm.corn_per_row + farm.potato_rows * farm.potatoes_per_row) / 2) :
  farm.potato_rows = 2 := by
  sorry


end potato_rows_count_l596_59686


namespace parkway_elementary_soccer_l596_59615

theorem parkway_elementary_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) (boys_soccer_percentage : ℚ) :
  total_students = 420 →
  boys = 312 →
  soccer_players = 250 →
  boys_soccer_percentage = 82 / 100 →
  (total_students - boys) - (soccer_players - (boys_soccer_percentage * soccer_players).floor) = 63 := by
  sorry

end parkway_elementary_soccer_l596_59615


namespace books_added_by_marta_l596_59612

theorem books_added_by_marta (initial_books final_books : ℕ) 
  (h1 : initial_books = 38)
  (h2 : final_books = 48)
  (h3 : final_books ≥ initial_books) :
  final_books - initial_books = 10 := by
  sorry

end books_added_by_marta_l596_59612


namespace larger_number_proof_l596_59620

theorem larger_number_proof (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  Nat.gcd a b = 23 ∧
  ∃ (x y : ℕ), x * y = Nat.lcm a b ∧ x = 13 ∧ y = 14 →
  max a b = 322 := by
sorry

end larger_number_proof_l596_59620


namespace forum_total_posts_per_day_l596_59638

/-- Represents a question and answer forum --/
structure Forum where
  members : ℕ
  questionsPerHour : ℕ
  answerRatio : ℕ

/-- Calculates the total number of questions and answers posted in a day --/
def totalPostsPerDay (f : Forum) : ℕ :=
  let questionsPerDay := f.members * (f.questionsPerHour * 24)
  let answersPerDay := f.members * (f.questionsPerHour * f.answerRatio * 24)
  questionsPerDay + answersPerDay

/-- Theorem stating the total number of posts per day for the given forum --/
theorem forum_total_posts_per_day :
  ∃ (f : Forum), f.members = 200 ∧ f.questionsPerHour = 3 ∧ f.answerRatio = 3 ∧
  totalPostsPerDay f = 57600 :=
by
  sorry

end forum_total_posts_per_day_l596_59638


namespace square_root_of_25_squared_l596_59654

theorem square_root_of_25_squared : Real.sqrt 25 ^ 2 = 25 := by
  sorry

end square_root_of_25_squared_l596_59654


namespace five_balls_three_boxes_l596_59611

/-- The number of ways to place distinguishable balls into indistinguishable boxes -/
def place_balls_in_boxes (n_balls : ℕ) (n_boxes : ℕ) : ℕ :=
  ((n_boxes ^ n_balls - n_boxes.choose 1 * (n_boxes - 1) ^ n_balls + n_boxes.choose 2) / n_boxes.factorial)

/-- Theorem: There are 25 ways to place 5 distinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : place_balls_in_boxes 5 3 = 25 := by
  sorry

end five_balls_three_boxes_l596_59611


namespace unique_solution_inequality_holds_max_value_l596_59688

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|
def h (a x : ℝ) : ℝ := |f x| + g a x

-- Theorem for part (1)
theorem unique_solution (a : ℝ) :
  (∃! x, |f x| = g a x) ↔ a < 0 :=
sorry

-- Theorem for part (2)
theorem inequality_holds (a : ℝ) :
  (∀ x, f x ≥ g a x) ↔ a ≤ -2 :=
sorry

-- Theorem for part (3)
theorem max_value (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) 2, h a x ≤ 
    if a ≥ 0 then 3*a + 3
    else if a ≥ -3 then a + 3
    else 0) ∧
  (∃ x ∈ Set.Icc (-2) 2, h a x = 
    if a ≥ 0 then 3*a + 3
    else if a ≥ -3 then a + 3
    else 0) :=
sorry

end unique_solution_inequality_holds_max_value_l596_59688


namespace sum_of_ratios_bound_l596_59645

theorem sum_of_ratios_bound (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  a / (b + c^2) + b / (c + a^2) + c / (a + b^2) ≥ 9/4 := by
  sorry

end sum_of_ratios_bound_l596_59645


namespace prism_volume_30_l596_59669

/-- A right rectangular prism with integer edge lengths -/
structure RightRectangularPrism where
  a : ℕ
  b : ℕ
  h : ℕ

/-- The volume of a right rectangular prism -/
def volume (p : RightRectangularPrism) : ℕ := p.a * p.b * p.h

/-- The areas of the faces of a right rectangular prism -/
def face_areas (p : RightRectangularPrism) : Finset ℕ :=
  {p.a * p.b, p.a * p.h, p.b * p.h}

theorem prism_volume_30 (p : RightRectangularPrism) :
  30 ∈ face_areas p → 13 ∈ face_areas p → volume p = 30 := by
  sorry

#check prism_volume_30

end prism_volume_30_l596_59669


namespace problem_solution_l596_59663

theorem problem_solution (a : ℝ) (h1 : a > 0) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + 9)
  (g : ℝ → ℝ) (hg : ∀ x, g x = x^2 - 5)
  (h2 : f (g a) = 25) : a = 3 := by
  sorry

end problem_solution_l596_59663


namespace total_pills_taken_l596_59698

-- Define the given conditions
def dose_mg : ℕ := 1000
def dose_interval_hours : ℕ := 6
def treatment_weeks : ℕ := 2
def mg_per_pill : ℕ := 500
def hours_per_day : ℕ := 24
def days_per_week : ℕ := 7

-- Define the theorem
theorem total_pills_taken : 
  (dose_mg / mg_per_pill) * 
  (hours_per_day / dose_interval_hours) * 
  (treatment_weeks * days_per_week) = 112 := by
sorry

end total_pills_taken_l596_59698


namespace min_value_of_f_l596_59667

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + Real.exp (-x)

theorem min_value_of_f (a : ℝ) :
  (∃ k : ℝ, k * f a 0 + 1 = 0 ∧ k * (a - 1) = -1) →
  (∃ x : ℝ, ∀ y : ℝ, f a y ≥ f a x) ∧
  (∃ x : ℝ, f a x = 4) :=
by sorry

end min_value_of_f_l596_59667


namespace quadratic_roots_condition_l596_59665

theorem quadratic_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 
   x₁^2 - (m+2)*x₁ + 1 = 0 ∧ 
   x₂^2 - (m+2)*x₂ + 1 = 0 ∧ 
   x₁ ≠ x₂) →
  m ≥ 0 := by
sorry

end quadratic_roots_condition_l596_59665


namespace sum_of_squares_l596_59684

-- Define the triangle FAC
structure Triangle :=
  (F A C : ℝ × ℝ)

-- Define the property of right angle FAC
def isRightAngle (t : Triangle) : Prop :=
  -- This is a placeholder for the right angle condition
  sorry

-- Define the length of CF
def CF_length (t : Triangle) : ℝ := 12

-- Define the area of square ACDE
def area_ACDE (t : Triangle) : ℝ :=
  let (_, A) := t.A
  let (_, C) := t.C
  (A - C) ^ 2

-- Define the area of square AFGH
def area_AFGH (t : Triangle) : ℝ :=
  let (F, _) := t.F
  let (A, _) := t.A
  (F - A) ^ 2

-- The theorem to be proved
theorem sum_of_squares (t : Triangle) 
  (h1 : isRightAngle t) 
  (h2 : CF_length t = 12) : 
  area_ACDE t + area_AFGH t = 144 :=
sorry

end sum_of_squares_l596_59684


namespace car_fuel_efficiency_l596_59629

theorem car_fuel_efficiency 
  (x : ℝ) 
  (h1 : x > 0) 
  (tank_capacity : ℝ) 
  (h2 : tank_capacity = 12) 
  (efficiency_improvement : ℝ) 
  (h3 : efficiency_improvement = 0.8) 
  (distance_increase : ℝ) 
  (h4 : distance_increase = 96) :
  (tank_capacity * x / efficiency_improvement) - (tank_capacity * x) = distance_increase →
  x = 32 :=
by sorry

end car_fuel_efficiency_l596_59629


namespace grid_selection_count_l596_59694

theorem grid_selection_count : ℕ := by
  -- Define the size of the grid
  let n : ℕ := 6
  
  -- Define the number of blocks to select
  let k : ℕ := 4
  
  -- Define the function to calculate combinations
  let choose (n m : ℕ) := Nat.choose n m
  
  -- Define the total number of combinations
  let total_combinations := choose n k * choose n k * Nat.factorial k
  
  -- Prove that the total number of combinations is 5400
  sorry

end grid_selection_count_l596_59694


namespace quadratic_function_properties_l596_59642

/-- A quadratic function f(x) = ax^2 + bx + 6 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 6

theorem quadratic_function_properties (a b : ℝ) :
  f a b 1 = 8 ∧ f a b (-1) = f a b 3 →
  (a + b = 2 ∧ f a b 2 = 6) := by
  sorry

end quadratic_function_properties_l596_59642


namespace sum_of_pentagon_angles_l596_59664

theorem sum_of_pentagon_angles : ∀ (A B C D E : ℝ),
  A + B + C + D + E = 180 * 3 :=
by
  sorry

end sum_of_pentagon_angles_l596_59664


namespace modulus_of_z_l596_59624

theorem modulus_of_z (z : ℂ) (h : z / (1 + 2 * I) = 1) : Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_z_l596_59624


namespace x_fourth_minus_reciprocal_fourth_l596_59658

theorem x_fourth_minus_reciprocal_fourth (x : ℝ) (h : x^2 - Real.sqrt 6 * x + 1 = 0) :
  |x^4 - 1/x^4| = 4 * Real.sqrt 2 := by
  sorry

end x_fourth_minus_reciprocal_fourth_l596_59658


namespace unique_intersection_values_l596_59601

-- Define the complex plane
variable (z : ℂ)

-- Define the condition from the original problem
def intersection_condition (k : ℝ) : Prop :=
  ∃! z : ℂ, Complex.abs (z - 4) = 3 * Complex.abs (z + 4) ∧ Complex.abs z = k

-- State the theorem
theorem unique_intersection_values :
  ∀ k : ℝ, intersection_condition k ↔ (k = 13 - Real.sqrt 153 ∨ k = 13 + Real.sqrt 153) :=
sorry

end unique_intersection_values_l596_59601


namespace skill_players_count_l596_59618

/-- Represents the football team's water consumption scenario -/
structure FootballTeam where
  cooler_capacity : ℕ
  num_linemen : ℕ
  lineman_consumption : ℕ
  skill_player_consumption : ℕ
  waiting_skill_players : ℕ

/-- Calculates the number of skill position players on the team -/
def num_skill_players (team : FootballTeam) : ℕ :=
  let remaining_water := team.cooler_capacity - team.num_linemen * team.lineman_consumption
  let drinking_skill_players := remaining_water / team.skill_player_consumption
  drinking_skill_players + team.waiting_skill_players

/-- Theorem stating the number of skill position players on the team -/
theorem skill_players_count (team : FootballTeam) 
  (h1 : team.cooler_capacity = 126)
  (h2 : team.num_linemen = 12)
  (h3 : team.lineman_consumption = 8)
  (h4 : team.skill_player_consumption = 6)
  (h5 : team.waiting_skill_players = 5) :
  num_skill_players team = 10 := by
  sorry

#eval num_skill_players {
  cooler_capacity := 126,
  num_linemen := 12,
  lineman_consumption := 8,
  skill_player_consumption := 6,
  waiting_skill_players := 5
}

end skill_players_count_l596_59618


namespace baseAngle_eq_pi_div_k_l596_59613

/-- An isosceles trapezoid inscribed around a circle -/
structure IsoscelesTrapezoidAroundCircle where
  /-- The ratio of the parallel sides -/
  k : ℝ
  /-- The angle at the base -/
  baseAngle : ℝ

/-- Theorem: The angle at the base of an isosceles trapezoid inscribed around a circle
    is equal to π/k, where k is the ratio of the parallel sides -/
theorem baseAngle_eq_pi_div_k (t : IsoscelesTrapezoidAroundCircle) :
  t.baseAngle = π / t.k :=
sorry

end baseAngle_eq_pi_div_k_l596_59613


namespace equation_solution_l596_59641

theorem equation_solution : ∃ x : ℝ, 
  (45 * x) / (3/4) = (37.5/100) * 1500 - (62.5/100) * 800 ∧ 
  abs (x - 1.0417) < 0.0001 := by
  sorry

end equation_solution_l596_59641


namespace solve_equation_l596_59657

theorem solve_equation :
  let y := 45 / (8 - 3/7)
  y = 315/53 := by
sorry

end solve_equation_l596_59657


namespace quadratic_equation_roots_l596_59696

/-- Given a quadratic equation ax^2 + bx + c = 0 with no real roots,
    if there exist two possible misinterpretations of the equation
    such that one yields roots 2 and 4, and the other yields roots -1 and 4,
    then (2b + 3c) / a = 12 -/
theorem quadratic_equation_roots (a b c : ℝ) : 
  (∀ x : ℝ, a * x^2 + b * x + c ≠ 0) →  -- No real roots
  (∃ a' : ℝ, a' * 4^2 + b * 4 + c = 0 ∧ a' * 2^2 + b * 2 + c = 0) →  -- Misinterpretation 1
  (∃ b' : ℝ, a * 4^2 + b' * 4 + c = 0 ∧ a * (-1)^2 + b' * (-1) + c = 0) →  -- Misinterpretation 2
  (2 * b + 3 * c) / a = 12 := by
sorry

end quadratic_equation_roots_l596_59696


namespace janet_dresses_l596_59699

/-- The number of dresses Janet has -/
def total_dresses : ℕ := 24

/-- The number of pockets in Janet's dresses -/
def total_pockets : ℕ := 32

theorem janet_dresses :
  (total_dresses / 2 / 3 * 2 + total_dresses / 2 * 2 / 3 * 3 = total_pockets) ∧
  (total_dresses > 0) := by
  sorry

end janet_dresses_l596_59699


namespace condition_necessary_not_sufficient_l596_59637

theorem condition_necessary_not_sufficient (a b : ℝ) :
  (a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧
  ¬(a ≠ 1 ∨ b ≠ 2 → a + b ≠ 3) := by
  sorry

end condition_necessary_not_sufficient_l596_59637


namespace sqrt_neg_three_l596_59693

theorem sqrt_neg_three (z : ℂ) : z * z = -3 ↔ z = Complex.I * Real.sqrt 3 ∨ z = -Complex.I * Real.sqrt 3 := by
  sorry

end sqrt_neg_three_l596_59693


namespace smallest_n_for_exact_tax_l596_59655

theorem smallest_n_for_exact_tax : ∃ (x : ℕ+), (104 * x : ℚ) / 10000 = 13 ∧
  ∀ (n : ℕ+), n < 13 → ¬∃ (y : ℕ+), (104 * y : ℚ) / 10000 = n := by
  sorry

end smallest_n_for_exact_tax_l596_59655


namespace shirt_cost_calculation_l596_59602

def shorts_cost : ℚ := 13.99
def jacket_cost : ℚ := 7.43
def total_cost : ℚ := 33.56

theorem shirt_cost_calculation :
  total_cost - (shorts_cost + jacket_cost) = 12.14 := by
  sorry

end shirt_cost_calculation_l596_59602


namespace scientific_notation_78922_l596_59648

theorem scientific_notation_78922 : 
  78922 = 7.8922 * (10 : ℝ)^4 := by sorry

end scientific_notation_78922_l596_59648


namespace constant_function_invariant_l596_59685

/-- Given a function f that is constant 3 for all real inputs, 
    prove that f(x + 5) = 3 for any real x -/
theorem constant_function_invariant (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = 3) :
  ∀ x : ℝ, f (x + 5) = 3 := by
  sorry

end constant_function_invariant_l596_59685


namespace divisor_product_theorem_l596_59616

/-- d(n) is the number of positive divisors of n -/
def d (n : ℕ) : ℕ := (Nat.divisors n).card

/-- s(n) is the sum of positive divisors of n -/
def s (n : ℕ) : ℕ := (Nat.divisors n).sum id

/-- The main theorem: s(x) * d(x) = 96 if and only if x is 14, 15, or 47 -/
theorem divisor_product_theorem (x : ℕ) : s x * d x = 96 ↔ x = 14 ∨ x = 15 ∨ x = 47 := by
  sorry

end divisor_product_theorem_l596_59616


namespace intersection_of_sets_l596_59628

open Set

theorem intersection_of_sets : 
  let A : Set ℝ := {x | 1 < x ∧ x < 8}
  let B : Set ℝ := {1, 3, 5, 6, 7}
  A ∩ B = {3, 5, 6, 7} := by
sorry

end intersection_of_sets_l596_59628


namespace proportional_enlargement_l596_59692

/-- Proportional enlargement of a rectangle -/
theorem proportional_enlargement (original_width original_height new_width : ℝ) 
  (h1 : original_width > 0)
  (h2 : original_height > 0)
  (h3 : new_width > 0) :
  let scale_factor := new_width / original_width
  let new_height := original_height * scale_factor
  (original_width = 3 ∧ original_height = 2 ∧ new_width = 12) → new_height = 8 := by
sorry

end proportional_enlargement_l596_59692


namespace triangle_problem_l596_59607

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Given conditions
  (b * Real.cos B = (a * Real.cos C + c * Real.cos A) / 2) →
  (a + c = Real.sqrt 10) →
  (b = 2) →
  -- Conclusions
  (B = π / 3) ∧
  (1/2 * a * c * Real.sin B = Real.sqrt 3 / 2) :=
by sorry

end triangle_problem_l596_59607


namespace min_side_length_triangle_l596_59608

theorem min_side_length_triangle (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∃ (h1 h2 h3 : ℝ), h1 > 0 ∧ h2 > 0 ∧ h3 > 0 ∧
    h1 * a = h2 * b ∧ h2 * b = h3 * c ∧
    h1 = 3 ∧ h2 = 4 ∧ h3 = 5) →
  min a (min b c) ≥ 12 :=
by sorry

end min_side_length_triangle_l596_59608


namespace sqrt_four_minus_one_l596_59691

theorem sqrt_four_minus_one : Real.sqrt 4 - 1 = 1 := by
  sorry

end sqrt_four_minus_one_l596_59691


namespace square_and_circle_l596_59656

theorem square_and_circle (square_area : ℝ) (side_length : ℝ) (circle_radius : ℝ) : 
  square_area = 1 →
  side_length ^ 2 = square_area →
  circle_radius * 2 = side_length →
  side_length = 1 ∧ circle_radius = 0.5 := by
  sorry

end square_and_circle_l596_59656


namespace original_number_proof_l596_59689

theorem original_number_proof (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 30 →
  (a + b + c + 50) / 4 = 40 →
  d = 10 := by
sorry

end original_number_proof_l596_59689


namespace ab_value_l596_59668

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end ab_value_l596_59668


namespace smallest_terminating_decimal_l596_59672

/-- A positive integer n such that n/(n+51) is a terminating decimal -/
def is_terminating_decimal (n : ℕ+) : Prop :=
  ∃ (a b : ℕ), n.val / (n.val + 51) = (a : ℚ) / (10^b : ℚ)

/-- 74 is the smallest positive integer n such that n/(n+51) is a terminating decimal -/
theorem smallest_terminating_decimal :
  (∀ m : ℕ+, m.val < 74 → ¬is_terminating_decimal m) ∧ is_terminating_decimal 74 :=
sorry

end smallest_terminating_decimal_l596_59672


namespace arithmetic_sequence_product_l596_59682

theorem arithmetic_sequence_product (b : ℕ → ℕ) : 
  (∀ n, b n < b (n + 1)) →  -- increasing sequence
  (∃ d : ℕ, ∀ n, b (n + 1) = b n + d) →  -- arithmetic sequence
  b 3 * b 4 = 72 → 
  b 2 * b 5 = 70 := by
sorry

end arithmetic_sequence_product_l596_59682


namespace even_odd_product_sum_zero_l596_59625

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- A function g: ℝ → ℝ is odd if g(-x) = -g(x) for all x ∈ ℝ -/
def IsOdd (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

/-- For even function f and odd function g, f(-x)g(-x) + f(x)g(x) = 0 for all x ∈ ℝ -/
theorem even_odd_product_sum_zero (f g : ℝ → ℝ) (hf : IsEven f) (hg : IsOdd g) :
    ∀ x : ℝ, f (-x) * g (-x) + f x * g x = 0 := by
  sorry

end even_odd_product_sum_zero_l596_59625


namespace rook_placements_l596_59683

def chessboard_size : ℕ := 8
def num_rooks : ℕ := 3

theorem rook_placements : 
  (chessboard_size.choose num_rooks) * (chessboard_size * (chessboard_size - 1) * (chessboard_size - 2)) = 18816 :=
by sorry

end rook_placements_l596_59683


namespace latus_rectum_for_parabola_l596_59631

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := y = -1/6 * x^2

/-- The equation of the latus rectum -/
def latus_rectum_equation (y : ℝ) : Prop := y = 3/2

/-- Theorem: The latus rectum equation for the given parabola -/
theorem latus_rectum_for_parabola :
  ∀ x y : ℝ, parabola_equation x y → latus_rectum_equation y :=
by sorry

end latus_rectum_for_parabola_l596_59631


namespace subset_implies_m_values_l596_59675

def A (m : ℝ) : Set ℝ := {-1, 3, 2*m-1}
def B (m : ℝ) : Set ℝ := {3, m}

theorem subset_implies_m_values (m : ℝ) :
  B m ⊆ A m → m = 1 ∨ m = -1 :=
by sorry

end subset_implies_m_values_l596_59675


namespace percentage_male_students_l596_59639

theorem percentage_male_students 
  (T : ℝ) -- Total number of students
  (M : ℝ) -- Number of male students
  (F : ℝ) -- Number of female students
  (h1 : M + F = T) -- Total students equation
  (h2 : (2/7) * M + (1/3) * F = 0.3 * T) -- Married students equation
  : M / T = 0.7 := by sorry

end percentage_male_students_l596_59639


namespace frank_remaining_money_l596_59670

def cheapest_lamp : ℕ := 20
def expensive_multiplier : ℕ := 3
def frank_money : ℕ := 90

theorem frank_remaining_money :
  frank_money - (cheapest_lamp * expensive_multiplier) = 30 := by
  sorry

end frank_remaining_money_l596_59670


namespace problem_solution_l596_59679

noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x

theorem problem_solution :
  (f (25 * Real.pi / 6) = 0) ∧
  (∀ α : ℝ, 0 < α ∧ α < Real.pi →
    f (α / 2) = 1 / 4 - Real.sqrt 3 / 2 →
    Real.sin α = (1 + 3 * Real.sqrt 5) / 8) :=
by sorry

end problem_solution_l596_59679


namespace expression_evaluation_l596_59610

theorem expression_evaluation :
  (3^1010 + 4^1012)^2 - (3^1010 - 4^1012)^2 = 10^2630 * 10^1012 :=
by sorry

end expression_evaluation_l596_59610


namespace sum_of_sequences_l596_59600

def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_of_sequences : 
  (arithmetic_sum 2 10 5) + (arithmetic_sum 10 10 5) = 260 := by
  sorry

end sum_of_sequences_l596_59600


namespace zongzi_purchase_l596_59614

/-- Represents the unit price and quantity of zongzi -/
structure Zongzi where
  price : ℝ
  quantity : ℝ

/-- Represents the purchase of zongzi -/
def Purchase (a b : Zongzi) : Prop :=
  a.price * a.quantity = 1500 ∧
  b.price * b.quantity = 1000 ∧
  b.quantity = a.quantity + 50 ∧
  a.price = 2 * b.price

/-- Represents the additional purchase constraint -/
def AdditionalPurchase (a b : Zongzi) (x : ℝ) : Prop :=
  x + (200 - x) = 200 ∧
  a.price * x + b.price * (200 - x) ≤ 1450

/-- Main theorem -/
theorem zongzi_purchase (a b : Zongzi) (x : ℝ) 
  (h1 : Purchase a b) (h2 : AdditionalPurchase a b x) : 
  b.price = 5 ∧ a.price = 10 ∧ x ≤ 90 := by
  sorry

#check zongzi_purchase

end zongzi_purchase_l596_59614


namespace sufficient_not_necessary_condition_l596_59674

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 4 → x ≥ 4) ∧ (∃ x, x ≥ 4 ∧ ¬(x > 4)) :=
by sorry

end sufficient_not_necessary_condition_l596_59674


namespace problems_per_page_l596_59673

theorem problems_per_page
  (total_problems : ℕ)
  (finished_problems : ℕ)
  (remaining_pages : ℕ)
  (h1 : total_problems = 60)
  (h2 : finished_problems = 20)
  (h3 : remaining_pages = 5)
  (h4 : remaining_pages > 0)
  : (total_problems - finished_problems) / remaining_pages = 8 := by
  sorry

end problems_per_page_l596_59673


namespace geometric_progression_first_term_l596_59678

theorem geometric_progression_first_term
  (S : ℝ)
  (sum_first_two : ℝ)
  (h1 : S = 15)
  (h2 : sum_first_two = 10) :
  ∃ (a : ℝ), (a = (15 * (Real.sqrt 3 - 1)) / Real.sqrt 3 ∨
              a = (15 * (Real.sqrt 3 + 1)) / Real.sqrt 3) ∧
             (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a + a * r) :=
by sorry

end geometric_progression_first_term_l596_59678


namespace square_to_rectangle_l596_59659

theorem square_to_rectangle (s : ℝ) (h1 : s > 0) 
  (h2 : s * (s + 3) - s * s = 18) : 
  s * s = 36 ∧ s * (s + 3) = 54 := by
  sorry

#check square_to_rectangle

end square_to_rectangle_l596_59659


namespace equations_equivalence_l596_59643

-- Define the function types
variable {X : Type} [Nonempty X]
variable (f₁ f₂ f₃ f₄ : X → ℝ)

-- Define the equations
def eq1 (x : X) := f₁ x / f₂ x = f₃ x / f₄ x
def eq2 (x : X) := f₁ x / f₂ x = (f₁ x + f₃ x) / (f₂ x + f₄ x)

-- Define the conditions
def cond1 (x : X) := eq1 f₁ f₂ f₃ f₄ x → f₂ x + f₄ x ≠ 0
def cond2 (x : X) := eq2 f₁ f₂ f₃ f₄ x → f₄ x ≠ 0

-- State the theorem
theorem equations_equivalence :
  (∀ x, eq1 f₁ f₂ f₃ f₄ x ↔ eq2 f₁ f₂ f₃ f₄ x) ↔
  (∀ x, cond1 f₁ f₂ f₃ f₄ x ∧ cond2 f₁ f₂ f₃ f₄ x) :=
sorry

end equations_equivalence_l596_59643


namespace cindy_paint_area_l596_59603

/-- Given that Allen, Ben, and Cindy are painting a fence, where:
    1) The ratio of work done by Allen : Ben : Cindy is 3 : 5 : 2
    2) The total fence area to be painted is 300 square feet
    Prove that Cindy paints 60 square feet of the fence. -/
theorem cindy_paint_area (total_area : ℝ) (allen_ratio ben_ratio cindy_ratio : ℕ) :
  total_area = 300 ∧ 
  allen_ratio = 3 ∧ 
  ben_ratio = 5 ∧ 
  cindy_ratio = 2 →
  cindy_ratio * (total_area / (allen_ratio + ben_ratio + cindy_ratio)) = 60 :=
by sorry

end cindy_paint_area_l596_59603


namespace consecutive_numbers_sum_l596_59666

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) := by
  sorry

end consecutive_numbers_sum_l596_59666


namespace distance_midpoint_endpoint_l596_59671

theorem distance_midpoint_endpoint (t : ℝ) : 
  let A : ℝ × ℝ := (t - 4, -1)
  let B : ℝ × ℝ := (-2, t + 3)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ((midpoint.1 - A.1)^2 + (midpoint.2 - A.2)^2 = t^2 / 2) →
  t = -5 := by
sorry

end distance_midpoint_endpoint_l596_59671


namespace max_value_x_sqrt_3_minus_x_squared_l596_59617

theorem max_value_x_sqrt_3_minus_x_squared :
  ∀ x : ℝ, 0 < x → x < Real.sqrt 3 →
    x * Real.sqrt (3 - x^2) ≤ 9/4 ∧
    ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < Real.sqrt 3 ∧ x₀ * Real.sqrt (3 - x₀^2) = 9/4 :=
by sorry

end max_value_x_sqrt_3_minus_x_squared_l596_59617


namespace workshop_average_salary_l596_59619

/-- Proves that the average salary of all workers in a workshop is 8000 rupees. -/
theorem workshop_average_salary :
  let total_workers : ℕ := 21
  let technicians : ℕ := 7
  let technician_salary : ℕ := 12000
  let non_technician_salary : ℕ := 6000
  (total_workers * (technicians * technician_salary + (total_workers - technicians) * non_technician_salary)) / (total_workers * total_workers) = 8000 := by
  sorry

end workshop_average_salary_l596_59619


namespace triangle_max_area_l596_59635

theorem triangle_max_area (a b c A B C : ℝ) :
  a = 2 →
  (2 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C →
  (∀ b' c' A' B' C',
    a = 2 →
    (2 + b') * (Real.sin A' - Real.sin B') = (c' - b') * Real.sin C' →
    a * b' * Real.sin C' / 2 ≤ Real.sqrt 3) →
  a * b * Real.sin C / 2 = Real.sqrt 3 :=
by sorry

end triangle_max_area_l596_59635


namespace workshop_average_age_l596_59647

theorem workshop_average_age (total_members : ℕ) (overall_avg : ℝ) 
  (num_women num_men num_speakers : ℕ) (women_avg men_avg : ℝ) : 
  total_members = 50 →
  overall_avg = 22 →
  num_women = 25 →
  num_men = 20 →
  num_speakers = 5 →
  women_avg = 20 →
  men_avg = 25 →
  (total_members : ℝ) * overall_avg = 
    (num_women : ℝ) * women_avg + (num_men : ℝ) * men_avg + (num_speakers : ℝ) * ((total_members : ℝ) * overall_avg - (num_women : ℝ) * women_avg - (num_men : ℝ) * men_avg) / (num_speakers : ℝ) →
  ((total_members : ℝ) * overall_avg - (num_women : ℝ) * women_avg - (num_men : ℝ) * men_avg) / (num_speakers : ℝ) = 20 :=
by sorry

end workshop_average_age_l596_59647


namespace h_composition_equals_902_l596_59604

/-- The function h as defined in the problem -/
def h (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

/-- Theorem stating that h(h(2)) = 902 -/
theorem h_composition_equals_902 : h (h 2) = 902 := by
  sorry

end h_composition_equals_902_l596_59604


namespace sandwich_combinations_l596_59652

theorem sandwich_combinations (meat_types : ℕ) (cheese_types : ℕ) : meat_types = 12 → cheese_types = 8 → (meat_types.choose 2) * cheese_types = 528 := by
  sorry

end sandwich_combinations_l596_59652


namespace no_zeros_of_g_l596_59681

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (hf_cont : Continuous f)
variable (hf_diff : Differentiable ℝ f)
variable (hf_pos : ∀ x, x * (deriv f x) + f x > 0)

-- Define the function g
def g (x : ℝ) := x * f x + 1

-- State the theorem
theorem no_zeros_of_g :
  ∀ x > 0, g f x ≠ 0 :=
sorry

end no_zeros_of_g_l596_59681


namespace range_of_a_l596_59623

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_even : is_even_function f)
  (h_incr : increasing_on_nonpositive f)
  (h_ineq : f a ≥ f 2) :
  a ∈ Set.Icc (-2) 2 :=
sorry

end range_of_a_l596_59623


namespace M_intersect_N_l596_59633

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem M_intersect_N : M ∩ N = {0, 2} := by sorry

end M_intersect_N_l596_59633


namespace dana_wins_l596_59677

/-- Represents a player in the game -/
inductive Player
  | Carl
  | Dana
  | Leah

/-- Represents the state of the game -/
structure GameState where
  chosenNumbers : List ℝ
  currentPlayer : Player

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : ℝ) : Prop :=
  0 ≤ move ∧ move ≤ 10 ∧
  ∀ n ∈ state.chosenNumbers, |move - n| ≥ 2

/-- Defines the next player in the turn order -/
def nextPlayer : Player → Player
  | Player.Carl => Player.Dana
  | Player.Dana => Player.Leah
  | Player.Leah => Player.Carl

/-- Represents a winning strategy for a player -/
def hasWinningStrategy (player : Player) : Prop :=
  ∀ initialState : GameState,
    initialState.currentPlayer = player →
    ∃ (strategy : GameState → ℝ),
      ∀ gameSequence : List ℝ,
        (∀ move ∈ gameSequence, isValidMove initialState move) →
        (∃ finalState : GameState,
          finalState.chosenNumbers = initialState.chosenNumbers ++ gameSequence ∧
          finalState.currentPlayer = player ∧
          ¬∃ move, isValidMove finalState move)

/-- The main theorem stating that Dana has a winning strategy -/
theorem dana_wins : hasWinningStrategy Player.Dana := by
  sorry

end dana_wins_l596_59677


namespace original_male_count_l596_59640

/-- Represents the number of students of each gender -/
structure StudentCount where
  male : ℕ
  female : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (s : StudentCount) : Prop :=
  (s.male : ℚ) / ((s.female : ℚ) - 15) = 2 ∧
  ((s.male : ℚ) - 45) / ((s.female : ℚ) - 15) = 1/5

/-- The theorem stating that the original number of male students is 50 -/
theorem original_male_count (s : StudentCount) :
  satisfiesConditions s → s.male = 50 := by
  sorry


end original_male_count_l596_59640


namespace min_additional_coins_for_alex_l596_59653

/-- The minimum number of additional coins needed for distribution -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed -/
theorem min_additional_coins_for_alex : 
  min_additional_coins 15 63 = 57 := by
  sorry

end min_additional_coins_for_alex_l596_59653


namespace solution_set_f_geq_3_range_of_a_l596_59634

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3/2} ∪ {x : ℝ | x ≥ 3/2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x > a^2 - x^2 + 2*x} = {a : ℝ | -Real.sqrt 5 < a ∧ a < Real.sqrt 5} := by sorry

end solution_set_f_geq_3_range_of_a_l596_59634


namespace quadratic_factorization_sum_l596_59662

theorem quadratic_factorization_sum (a b c d : ℤ) : 
  (∀ x : ℚ, 6 * x^2 + x - 12 = (a * x + b) * (c * x + d)) →
  |a| + |b| + |c| + |d| = 12 := by
  sorry

end quadratic_factorization_sum_l596_59662


namespace remainder_of_1725_base14_div_9_l596_59621

/-- Converts a base-14 number to decimal --/
def base14ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 14 + d) 0

/-- The base-14 representation of 1725₁₄ --/
def number : List Nat := [1, 7, 2, 5]

theorem remainder_of_1725_base14_div_9 :
  (base14ToDecimal number) % 9 = 0 := by
  sorry

end remainder_of_1725_base14_div_9_l596_59621


namespace unique_n_for_prime_power_difference_l596_59687

def is_power_of_three (x : ℕ) : Prop :=
  ∃ a : ℕ, x = 3^a ∧ a > 0

theorem unique_n_for_prime_power_difference :
  ∃! n : ℕ, n > 0 ∧ 
    (∃ p : ℕ, Nat.Prime p ∧ is_power_of_three (p^n - (p-1)^n)) :=
by
  sorry

end unique_n_for_prime_power_difference_l596_59687


namespace maria_has_nineteen_towels_l596_59661

/-- The number of towels Maria ended up with after shopping and giving some to her mother. -/
def marias_remaining_towels (green_towels white_towels given_to_mother : ℕ) : ℕ :=
  green_towels + white_towels - given_to_mother

/-- Theorem stating that Maria ended up with 19 towels. -/
theorem maria_has_nineteen_towels :
  marias_remaining_towels 40 44 65 = 19 := by
  sorry

end maria_has_nineteen_towels_l596_59661


namespace max_value_cosine_function_l596_59646

theorem max_value_cosine_function (x : ℝ) :
  ∃ (k : ℤ), 2 * Real.cos x - 1 ≤ 2 * Real.cos (2 * k * Real.pi) - 1 :=
by sorry

end max_value_cosine_function_l596_59646


namespace toms_blue_marbles_l596_59609

theorem toms_blue_marbles (jason_blue : ℕ) (total_blue : ℕ) 
  (h1 : jason_blue = 44)
  (h2 : total_blue = 68) :
  total_blue - jason_blue = 24 := by
sorry

end toms_blue_marbles_l596_59609


namespace james_payment_is_18_l596_59632

/-- James's meal cost -/
def james_meal : ℚ := 16

/-- Friend's meal cost -/
def friend_meal : ℚ := 14

/-- Tip percentage -/
def tip_percent : ℚ := 20 / 100

/-- Calculate James's payment given the meal costs and tip percentage -/
def calculate_james_payment (james_meal friend_meal tip_percent : ℚ) : ℚ :=
  let total_before_tip := james_meal + friend_meal
  let tip := tip_percent * total_before_tip
  let total_with_tip := total_before_tip + tip
  total_with_tip / 2

/-- Theorem stating that James's payment is $18 -/
theorem james_payment_is_18 :
  calculate_james_payment james_meal friend_meal tip_percent = 18 := by
  sorry

end james_payment_is_18_l596_59632


namespace inheritance_calculation_l596_59626

theorem inheritance_calculation (x : ℝ) 
  (h1 : x > 0)
  (h2 : 0.3 * x + 0.12 * (0.7 * x) + 0.05 * (0.7 * x - 0.12 * (0.7 * x)) = 16800) :
  x = 40500 := by
  sorry

end inheritance_calculation_l596_59626


namespace production_today_is_90_l596_59695

/-- Calculates the production for today given the previous average, new average, and number of previous days. -/
def todayProduction (prevAvg newAvg : ℚ) (prevDays : ℕ) : ℚ :=
  (newAvg * (prevDays + 1) : ℚ) - (prevAvg * prevDays : ℚ)

/-- Proves that the production today is 90 units, given the specified conditions. -/
theorem production_today_is_90 :
  todayProduction 60 62 14 = 90 := by
  sorry

#eval todayProduction 60 62 14

end production_today_is_90_l596_59695


namespace james_original_weight_l596_59636

/-- Proves that given the conditions of James's weight gain, his original weight was 120 kg -/
theorem james_original_weight :
  ∀ W : ℝ,
  W > 0 →
  let muscle_gain := 0.20 * W
  let fat_gain := 0.25 * muscle_gain
  let final_weight := W + muscle_gain + fat_gain
  final_weight = 150 →
  W = 120 := by
sorry

end james_original_weight_l596_59636


namespace batsman_average_after_12th_innings_l596_59630

/-- Represents a batsman's performance over a series of innings -/
structure BatsmanPerformance where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional innings -/
def newAverage (bp : BatsmanPerformance) (newScore : ℕ) : ℚ :=
  (bp.totalRuns + newScore) / (bp.innings + 1)

theorem batsman_average_after_12th_innings
  (bp : BatsmanPerformance)
  (h1 : bp.innings = 11)
  (h2 : newAverage bp 48 = bp.average + 2)
  : newAverage bp 48 = 26 := by
  sorry

end batsman_average_after_12th_innings_l596_59630


namespace remainder_proof_l596_59651

theorem remainder_proof (n : ℕ) (h1 : n = 129) (h2 : 1428 % n = 9) : 2206 % n = 13 := by
  sorry

end remainder_proof_l596_59651


namespace cookies_in_box_l596_59644

/-- The number of cookies Jackson's oldest son gets after school -/
def oldest_son_cookies : ℕ := 4

/-- The number of cookies Jackson's youngest son gets after school -/
def youngest_son_cookies : ℕ := 2

/-- The number of days a box of cookies lasts -/
def box_duration : ℕ := 9

/-- The total number of cookies in the box -/
def total_cookies : ℕ := oldest_son_cookies + youngest_son_cookies * box_duration

theorem cookies_in_box : total_cookies = 54 := by
  sorry

end cookies_in_box_l596_59644
