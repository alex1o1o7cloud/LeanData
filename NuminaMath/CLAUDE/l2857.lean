import Mathlib

namespace NUMINAMATH_CALUDE_ginos_bears_l2857_285702

/-- The number of brown bears Gino has -/
def brown_bears : ℕ := 15

/-- The number of white bears Gino has -/
def white_bears : ℕ := 24

/-- The number of black bears Gino has -/
def black_bears : ℕ := 27

/-- The number of polar bears Gino has -/
def polar_bears : ℕ := 12

/-- The number of grizzly bears Gino has -/
def grizzly_bears : ℕ := 18

/-- The total number of bears Gino has -/
def total_bears : ℕ := brown_bears + white_bears + black_bears + polar_bears + grizzly_bears

theorem ginos_bears : total_bears = 96 := by
  sorry

end NUMINAMATH_CALUDE_ginos_bears_l2857_285702


namespace NUMINAMATH_CALUDE_standard_pairs_parity_l2857_285787

/-- Represents a color of a square on the chessboard -/
inductive Color
| Red
| Blue

/-- Represents a chessboard of size m × n -/
structure Chessboard (m n : ℕ) where
  cells : Fin m → Fin n → Color
  m_ge_three : m ≥ 3
  n_ge_three : n ≥ 3

/-- Counts the number of blue cells on the border of the chessboard (excluding corners) -/
def countBlueBorderCells (board : Chessboard m n) : ℕ :=
  sorry

/-- Counts the number of "standard pairs" on the chessboard -/
def countStandardPairs (board : Chessboard m n) : ℕ :=
  sorry

/-- Theorem stating the relationship between the number of standard pairs and blue border cells -/
theorem standard_pairs_parity (m n : ℕ) (board : Chessboard m n) :
  Odd (countStandardPairs board) ↔ Odd (countBlueBorderCells board) :=
sorry

end NUMINAMATH_CALUDE_standard_pairs_parity_l2857_285787


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l2857_285708

/-- Given two circles and a line tangent to both, proves the x-coordinate of the intersection point --/
theorem tangent_line_intersection (r1 r2 c2_x : ℝ) (h1 : r1 = 2) (h2 : r2 = 7) (h3 : c2_x = 15) :
  ∃ x : ℝ, x > 0 ∧ (r1 / x = r2 / (c2_x - x)) ∧ x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l2857_285708


namespace NUMINAMATH_CALUDE_combination_simplification_l2857_285748

theorem combination_simplification (n : ℕ) : 
  (n.choose (n - 2)) + (n.choose 3) + ((n + 1).choose 2) = ((n + 2).choose 3) := by
  sorry

end NUMINAMATH_CALUDE_combination_simplification_l2857_285748


namespace NUMINAMATH_CALUDE_dolphin_training_l2857_285782

theorem dolphin_training (total : ℕ) (fully_trained_ratio : ℚ) (semi_trained_ratio : ℚ)
  (beginner_ratio : ℚ) (intermediate_ratio : ℚ)
  (h1 : total = 120)
  (h2 : fully_trained_ratio = 1/4)
  (h3 : semi_trained_ratio = 1/6)
  (h4 : beginner_ratio = 3/8)
  (h5 : intermediate_ratio = 5/9) :
  let fully_trained := (total : ℚ) * fully_trained_ratio
  let remaining_after_fully_trained := total - fully_trained.floor
  let semi_trained := (remaining_after_fully_trained : ℚ) * semi_trained_ratio
  let untrained := remaining_after_fully_trained - semi_trained.floor
  let semi_and_untrained := semi_trained.floor + untrained
  let in_beginner := (semi_and_untrained : ℚ) * beginner_ratio
  let remaining_after_beginner := semi_and_untrained - in_beginner.floor
  let start_intermediate := (remaining_after_beginner : ℚ) * intermediate_ratio
  start_intermediate.floor = 31 :=
by sorry

end NUMINAMATH_CALUDE_dolphin_training_l2857_285782


namespace NUMINAMATH_CALUDE_wall_length_l2857_285772

/-- The length of a rectangular wall with a trapezoidal mirror -/
theorem wall_length (a b h w : ℝ) (ha : a > 0) (hb : b > 0) (hh : h > 0) (hw : w > 0) :
  (a + b) * h / 2 * 2 = w * (580 / 27) →
  a = 34 →
  b = 24 →
  h = 20 →
  w = 54 →
  580 / 27 = 580 / 27 :=
by sorry

end NUMINAMATH_CALUDE_wall_length_l2857_285772


namespace NUMINAMATH_CALUDE_tourist_travel_speeds_l2857_285788

theorem tourist_travel_speeds (total_distance : ℝ) (car_fraction : ℚ) (speed_difference : ℝ) (time_difference : ℝ) :
  total_distance = 160 ∧
  car_fraction = 5/8 ∧
  speed_difference = 20 ∧
  time_difference = 1/4 →
  (∃ (car_speed boat_speed : ℝ),
    (car_speed = 80 ∧ boat_speed = 60) ∨
    (car_speed = 100 ∧ boat_speed = 80)) ∧
    (car_speed - boat_speed = speed_difference) ∧
    (total_distance * car_fraction / car_speed = 
     total_distance * (1 - car_fraction) / boat_speed + time_difference) :=
by sorry

end NUMINAMATH_CALUDE_tourist_travel_speeds_l2857_285788


namespace NUMINAMATH_CALUDE_animal_path_distance_l2857_285712

/-- The total distance traveled by an animal along a specific path between two concentric circles -/
theorem animal_path_distance (r₁ r₂ : ℝ) (h₁ : r₁ = 15) (h₂ : r₂ = 25) :
  let outer_arc := (1/4) * 2 * Real.pi * r₂
  let radial_line := r₂ - r₁
  let inner_circle := 2 * Real.pi * r₁
  outer_arc + radial_line + inner_circle + radial_line = 42.5 * Real.pi + 20 := by
  sorry

end NUMINAMATH_CALUDE_animal_path_distance_l2857_285712


namespace NUMINAMATH_CALUDE_range_of_a_l2857_285764

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + 2*x + a ≥ 0) ↔ a ∈ Set.Ici (-8) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2857_285764


namespace NUMINAMATH_CALUDE_triangles_congruent_l2857_285761

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

-- Define area and perimeter functions
def area (t : Triangle) : ℝ := sorry
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem triangles_congruent (t1 t2 : Triangle) 
  (h_area : area t1 = area t2)
  (h_perimeter : perimeter t1 = perimeter t2)
  (h_side : t1.a = t2.a) :
  t1 = t2 := by sorry

end NUMINAMATH_CALUDE_triangles_congruent_l2857_285761


namespace NUMINAMATH_CALUDE_heather_blocks_shared_l2857_285737

/-- The number of blocks Heather shared with Jose -/
def blocks_shared (initial final : ℕ) : ℕ := initial - final

/-- Theorem stating that the number of blocks shared is the difference between initial and final counts -/
theorem heather_blocks_shared : 
  blocks_shared 86 45 = 41 := by
  sorry

end NUMINAMATH_CALUDE_heather_blocks_shared_l2857_285737


namespace NUMINAMATH_CALUDE_pen_price_relationship_l2857_285736

/-- The relationship between the number of pens and their total price -/
theorem pen_price_relationship (x y : ℝ) : y = (3/2) * x ↔ 
  ∃ (boxes : ℝ), 
    x = 12 * boxes ∧ 
    y = 18 * boxes :=
by sorry

end NUMINAMATH_CALUDE_pen_price_relationship_l2857_285736


namespace NUMINAMATH_CALUDE_ball_pit_count_l2857_285768

theorem ball_pit_count : ∃ (total : ℕ), 
  let red := total / 4
  let non_red := total - red
  let blue := non_red / 5
  let neither_red_nor_blue := total - red - blue
  neither_red_nor_blue = 216 ∧ total = 360 := by
sorry

end NUMINAMATH_CALUDE_ball_pit_count_l2857_285768


namespace NUMINAMATH_CALUDE_segment_ratio_l2857_285759

/-- Given points E, F, G, and H on a line in that order, with specified distances between them,
    prove that the ratio of EG to FH is 9:17. -/
theorem segment_ratio (E F G H : ℝ) : 
  F - E = 3 →
  G - F = 6 →
  H - G = 4 →
  H - E = 20 →
  (G - E) / (H - F) = 9 / 17 := by
  sorry

end NUMINAMATH_CALUDE_segment_ratio_l2857_285759


namespace NUMINAMATH_CALUDE_terry_spending_l2857_285796

def weekly_spending (monday : ℚ) : ℚ :=
  let tuesday := 2 * monday
  let wednesday := 2 * (monday + tuesday)
  let thursday := (monday + tuesday + wednesday) / 3
  let friday := thursday - 4
  let saturday := friday + (friday / 2)
  let sunday := tuesday + saturday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

theorem terry_spending :
  weekly_spending 6 = 140 := by sorry

end NUMINAMATH_CALUDE_terry_spending_l2857_285796


namespace NUMINAMATH_CALUDE_spinner_probability_l2857_285797

theorem spinner_probability (p_A p_B p_C p_D p_E : ℚ) : 
  p_A = 1/5 →
  p_B = 1/10 →
  p_D = p_E →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_D = 7/20 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l2857_285797


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2857_285726

theorem polynomial_division_remainder (x : ℂ) : 
  (x^6 - 1) * (x^3 - 1) % (x^2 + x + 1) = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2857_285726


namespace NUMINAMATH_CALUDE_friends_total_earnings_l2857_285718

/-- The total earnings of four friends selling electronics on eBay -/
def total_earnings (lauryn_earnings : ℝ) : ℝ :=
  let aurelia_earnings := 0.7 * lauryn_earnings
  let jackson_earnings := 1.5 * aurelia_earnings
  let maya_earnings := 0.4 * jackson_earnings
  lauryn_earnings + aurelia_earnings + jackson_earnings + maya_earnings

/-- Theorem stating that the total earnings of the four friends is $6340 -/
theorem friends_total_earnings :
  total_earnings 2000 = 6340 := by
  sorry

end NUMINAMATH_CALUDE_friends_total_earnings_l2857_285718


namespace NUMINAMATH_CALUDE_HG_ratio_l2857_285763

-- Define the equation
def equation (G H x : ℝ) : Prop :=
  (G / (x + 7) + H / (x^2 - 6*x) = (x^2 - 3*x + 15) / (x^3 + x^2 - 42*x))

-- State the theorem
theorem HG_ratio (G H : ℤ) : 
  (∀ x : ℝ, x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 → equation G H x) →
  (H : ℝ) / (G : ℝ) = 15 / 7 :=
sorry

end NUMINAMATH_CALUDE_HG_ratio_l2857_285763


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l2857_285766

theorem root_difference_implies_k_value (k : ℝ) :
  (∃ r s : ℝ, r^2 + k*r + 12 = 0 ∧ s^2 + k*s + 12 = 0) →
  (∃ r s : ℝ, (r+7)^2 - k*(r+7) + 12 = 0 ∧ (s+7)^2 - k*(s+7) + 12 = 0) →
  (∀ r s : ℝ, r^2 + k*r + 12 = 0 ∧ s^2 + k*s + 12 = 0 →
              (r+7)^2 - k*(r+7) + 12 = 0 ∧ (s+7)^2 - k*(s+7) + 12 = 0) →
  k = 7 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l2857_285766


namespace NUMINAMATH_CALUDE_fifth_pile_magazines_l2857_285794

/-- Represents the number of magazines in each pile -/
def magazine_sequence : ℕ → ℕ
| 0 => 3  -- First pile (health)
| 1 => 4  -- Second pile (technology)
| 2 => 6  -- Third pile (fashion)
| 3 => 9  -- Fourth pile (travel)
| n + 4 => magazine_sequence (n + 3) + (n + 4)  -- Subsequent piles

/-- The theorem stating that the fifth pile will contain 13 magazines -/
theorem fifth_pile_magazines : magazine_sequence 4 = 13 := by
  sorry


end NUMINAMATH_CALUDE_fifth_pile_magazines_l2857_285794


namespace NUMINAMATH_CALUDE_linear_function_through_minus_one_zero_l2857_285760

/-- A linear function passing through (-1, 0) has slope 1 -/
theorem linear_function_through_minus_one_zero (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1) → 0 = k * (-1) + 1 → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_through_minus_one_zero_l2857_285760


namespace NUMINAMATH_CALUDE_swimming_pool_capacity_l2857_285756

theorem swimming_pool_capacity (initial_fraction : ℚ) (added_amount : ℚ) (final_fraction : ℚ) :
  initial_fraction = 1/3 →
  added_amount = 180 →
  final_fraction = 4/5 →
  ∃ (total_capacity : ℚ), 
    total_capacity * initial_fraction + added_amount = total_capacity * final_fraction ∧
    total_capacity = 2700/7 := by
  sorry

#eval (2700 : ℚ) / 7

end NUMINAMATH_CALUDE_swimming_pool_capacity_l2857_285756


namespace NUMINAMATH_CALUDE_cadence_worked_five_months_longer_l2857_285771

/-- Calculates the number of months longer Cadence worked at her new company --/
def months_longer_at_new_company (
  old_salary : ℕ)
  (salary_increase_percent : ℕ)
  (old_company_months : ℕ)
  (total_earnings : ℕ) : ℕ :=
  let new_salary := old_salary + (old_salary * salary_increase_percent) / 100
  let x := (total_earnings - old_salary * old_company_months) / new_salary - old_company_months
  x

/-- Proves that Cadence worked 5 months longer at her new company --/
theorem cadence_worked_five_months_longer :
  months_longer_at_new_company 5000 20 36 426000 = 5 := by
  sorry

#eval months_longer_at_new_company 5000 20 36 426000

end NUMINAMATH_CALUDE_cadence_worked_five_months_longer_l2857_285771


namespace NUMINAMATH_CALUDE_complement_union_A_B_l2857_285705

def A : Set ℝ := {x : ℝ | x ≤ 0}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

theorem complement_union_A_B : 
  (A ∪ B)ᶜ = {x : ℝ | x > 1} :=
sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l2857_285705


namespace NUMINAMATH_CALUDE_max_value_xy_l2857_285753

theorem max_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 1) :
  xy ≤ 1/12 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3*y₀ = 1 ∧ x₀*y₀ = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xy_l2857_285753


namespace NUMINAMATH_CALUDE_not_eight_sum_l2857_285792

theorem not_eight_sum (a b c : ℕ) (h : 2^a * 3^b * 4^c = 192) : a + b + c ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_not_eight_sum_l2857_285792


namespace NUMINAMATH_CALUDE_group_element_identity_l2857_285727

variables {G : Type*} [Group G]

theorem group_element_identity (g h : G) (n : ℕ) 
  (h1 : g * h * g = h * g^2 * h)
  (h2 : g^3 = 1)
  (h3 : h^n = 1)
  (h4 : Odd n) :
  h = 1 := by sorry

end NUMINAMATH_CALUDE_group_element_identity_l2857_285727


namespace NUMINAMATH_CALUDE_baron_munchausen_claim_false_l2857_285790

theorem baron_munchausen_claim_false : ∃ n : ℕ, 
  10 ≤ n ∧ n ≤ 99 ∧ ¬∃ m : ℕ, 0 ≤ m ∧ m ≤ 99 ∧ (100 * n + m)^2 = 100 * n + m := by
  sorry

end NUMINAMATH_CALUDE_baron_munchausen_claim_false_l2857_285790


namespace NUMINAMATH_CALUDE_no_solution_exists_l2857_285767

theorem no_solution_exists : ¬∃ (x : ℝ), Real.arccos (4/5) - Real.arccos (-4/5) = Real.arcsin x := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2857_285767


namespace NUMINAMATH_CALUDE_bamboo_pole_problem_l2857_285750

/-- 
Given a bamboo pole of height 10 feet, if the top part when bent to the ground 
reaches a point 3 feet from the base, then the length of the broken part is 109/20 feet.
-/
theorem bamboo_pole_problem (h : ℝ) (x : ℝ) (y : ℝ) :
  h = 10 ∧ 
  x + y = h ∧ 
  x^2 + 3^2 = y^2 →
  y = 109/20 := by
sorry

end NUMINAMATH_CALUDE_bamboo_pole_problem_l2857_285750


namespace NUMINAMATH_CALUDE_sets_satisfying_union_condition_l2857_285773

theorem sets_satisfying_union_condition :
  ∃! (S : Finset (Finset ℕ)), 
    (∀ A ∈ S, A ∪ {1, 2} = {1, 2, 3}) ∧ 
    (∀ A, A ∪ {1, 2} = {1, 2, 3} → A ∈ S) ∧
    Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_sets_satisfying_union_condition_l2857_285773


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l2857_285751

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x - 1

-- State the theorem
theorem root_exists_in_interval :
  (∀ x ∈ Set.Icc 0 0.5, ContinuousAt f x) →
  f 0 < 0 →
  f 0.5 > 0 →
  ∃ x ∈ Set.Ioo 0 0.5, f x = 0 := by
  sorry

#check root_exists_in_interval

end NUMINAMATH_CALUDE_root_exists_in_interval_l2857_285751


namespace NUMINAMATH_CALUDE_walking_speed_proof_l2857_285758

/-- The walking speed of a man who covers the same distance in 9 hours walking
    and in 3 hours running at 24 kmph. -/
def walking_speed : ℝ := 8

theorem walking_speed_proof :
  walking_speed * 9 = 24 * 3 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_proof_l2857_285758


namespace NUMINAMATH_CALUDE_optimal_discount_order_l2857_285741

def original_price : ℝ := 30
def flat_discount : ℝ := 5
def percentage_discount : ℝ := 0.25

theorem optimal_discount_order :
  (original_price * (1 - percentage_discount) - flat_discount) -
  (original_price - flat_discount) * (1 - percentage_discount) = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_optimal_discount_order_l2857_285741


namespace NUMINAMATH_CALUDE_simplify_expression_l2857_285775

theorem simplify_expression (x y : ℝ) :
  5 * x^4 + 3 * x^2 * y - 4 - 3 * x^2 * y - 3 * x^4 - 1 = 2 * x^4 - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2857_285775


namespace NUMINAMATH_CALUDE_danny_bottle_caps_wrappers_l2857_285713

theorem danny_bottle_caps_wrappers : 
  let bottle_caps_found : ℕ := 50
  let wrappers_found : ℕ := 46
  bottle_caps_found - wrappers_found = 4 := by
  sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_wrappers_l2857_285713


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2857_285716

theorem necessary_but_not_sufficient :
  ∀ a : ℝ,
  (∀ x : ℝ, x^2 + 1 > a) →
  (∃ b : ℝ, b > 0 ∧ b ≠ 1 ∧ (∀ x y : ℝ, x < y → b^x > b^y)) ∧
  (∃ c : ℝ, (∀ x : ℝ, x^2 + 1 > c) ∧ 
   ¬(∀ x y : ℝ, x < y → c^x > c^y)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2857_285716


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2857_285730

def is_divisible_by_one_of (F : ℤ → ℤ) (divisors : List ℤ) : Prop :=
  ∀ n : ℤ, ∃ a ∈ divisors, (F n) % a = 0

theorem polynomial_divisibility
  (F : ℤ → ℤ)
  (divisors : List ℤ)
  (h_polynomial : ∀ x y : ℤ, (F x - F y) % (x - y) = 0)
  (h_divisible : is_divisible_by_one_of F divisors) :
  ∃ a ∈ divisors, ∀ n : ℤ, (F n) % a = 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2857_285730


namespace NUMINAMATH_CALUDE_xyz_sum_l2857_285731

theorem xyz_sum (x y z : ℕ+) 
  (h : (x + y + z : ℕ+)^3 - x^3 - y^3 - z^3 = 300) : 
  (x : ℕ) + y + z = 7 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l2857_285731


namespace NUMINAMATH_CALUDE_max_third_place_books_l2857_285704

structure BookDistribution where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

def is_valid_distribution (d : BookDistribution) : Prop :=
  d.first > d.second ∧
  d.second > d.third ∧
  d.third > d.fourth ∧
  d.fourth > d.fifth ∧
  d.first % 100 = 0 ∧
  d.second % 100 = 0 ∧
  d.third % 100 = 0 ∧
  d.fourth % 100 = 0 ∧
  d.fifth % 100 = 0 ∧
  d.first = d.second + d.third ∧
  d.second = d.fourth + d.fifth ∧
  d.first + d.second + d.third + d.fourth + d.fifth ≤ 10000

theorem max_third_place_books :
  ∀ d : BookDistribution,
    is_valid_distribution d →
    d.third ≤ 1900 :=
by sorry

end NUMINAMATH_CALUDE_max_third_place_books_l2857_285704


namespace NUMINAMATH_CALUDE_original_solution_concentration_l2857_285720

/-- Represents a chemical solution with a certain concentration --/
structure ChemicalSolution :=
  (concentration : ℝ)

/-- Represents a mixture of two chemical solutions --/
def mix (s1 s2 : ChemicalSolution) (ratio : ℝ) : ChemicalSolution :=
  { concentration := ratio * s1.concentration + (1 - ratio) * s2.concentration }

/-- Theorem: If half of an original solution is replaced with a 60% solution,
    resulting in a 55% solution, then the original solution was 50% --/
theorem original_solution_concentration
  (original replacement result : ChemicalSolution)
  (h1 : replacement.concentration = 0.6)
  (h2 : result = mix original replacement 0.5)
  (h3 : result.concentration = 0.55) :
  original.concentration = 0.5 := by
  sorry

#check original_solution_concentration

end NUMINAMATH_CALUDE_original_solution_concentration_l2857_285720


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l2857_285739

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l2857_285739


namespace NUMINAMATH_CALUDE_complex_square_minus_i_l2857_285798

theorem complex_square_minus_i (z : ℂ) : z = 1 + I → z^2 - I = I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_minus_i_l2857_285798


namespace NUMINAMATH_CALUDE_range_of_a1_l2857_285784

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a (n + 1) = 2 * (|a n| - 1)

/-- The sequence is bounded by some positive constant M -/
def BoundedSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ n : ℕ+, |a n| ≤ M

/-- The main theorem stating the range of a₁ -/
theorem range_of_a1 (a : ℕ+ → ℝ) 
    (h1 : RecurrenceSequence a) 
    (h2 : BoundedSequence a) : 
    -2 ≤ a 1 ∧ a 1 ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_range_of_a1_l2857_285784


namespace NUMINAMATH_CALUDE_loaves_baked_l2857_285781

def flour_available : ℝ := 5
def flour_per_loaf : ℝ := 2.5

theorem loaves_baked : ⌊flour_available / flour_per_loaf⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_loaves_baked_l2857_285781


namespace NUMINAMATH_CALUDE_tangent_length_from_point_to_circle_l2857_285732

/-- The length of the tangent from a point to a circle -/
theorem tangent_length_from_point_to_circle 
  (P : ℝ × ℝ) -- Point P
  (center : ℝ × ℝ) -- Center of the circle
  (r : ℝ) -- Radius of the circle
  (h1 : P = (2, 3)) -- P coordinates
  (h2 : center = (0, 0)) -- Circle center
  (h3 : r = 1) -- Circle radius
  (h4 : (P.1 - center.1)^2 + (P.2 - center.2)^2 > r^2) -- P is outside the circle
  : Real.sqrt ((P.1 - center.1)^2 + (P.2 - center.2)^2 - r^2) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_length_from_point_to_circle_l2857_285732


namespace NUMINAMATH_CALUDE_rs_value_l2857_285755

theorem rs_value (r s : ℝ) (hr : 0 < r) (hs : 0 < s) 
  (h1 : r^2 + s^2 = 2) (h2 : r^4 + s^4 = 9/8) : r * s = Real.sqrt 23 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rs_value_l2857_285755


namespace NUMINAMATH_CALUDE_factorial_expression_equals_2884_l2857_285744

theorem factorial_expression_equals_2884 :
  (Real.sqrt ((Nat.factorial 5 * Nat.factorial 4) + 2^2))^2 = 2884 := by sorry

end NUMINAMATH_CALUDE_factorial_expression_equals_2884_l2857_285744


namespace NUMINAMATH_CALUDE_remove_four_for_target_average_l2857_285722

def original_list : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
def target_average : ℚ := 63/10

theorem remove_four_for_target_average :
  let remaining_list := original_list.filter (· ≠ 4)
  (remaining_list.sum : ℚ) / remaining_list.length = target_average := by
  sorry

end NUMINAMATH_CALUDE_remove_four_for_target_average_l2857_285722


namespace NUMINAMATH_CALUDE_c_share_of_profit_l2857_285700

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_share_of_profit (investment : ℕ) (total_investment : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * total_profit) / total_investment

theorem c_share_of_profit (investment_a investment_b investment_c total_profit : ℕ) 
  (h1 : investment_a = 27000)
  (h2 : investment_b = 72000)
  (h3 : investment_c = 81000)
  (h4 : total_profit = 80000) :
  calculate_share_of_profit investment_c (investment_a + investment_b + investment_c) total_profit = 36000 :=
by
  sorry

#eval calculate_share_of_profit 81000 (27000 + 72000 + 81000) 80000

end NUMINAMATH_CALUDE_c_share_of_profit_l2857_285700


namespace NUMINAMATH_CALUDE_mod_thirteen_five_eight_l2857_285795

theorem mod_thirteen_five_eight (m : ℕ) : 
  13^5 % 8 = m → 0 ≤ m → m < 8 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_thirteen_five_eight_l2857_285795


namespace NUMINAMATH_CALUDE_root_values_l2857_285728

theorem root_values (p q r s k : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) 
  (h1 : p * k^3 + q * k^2 + r * k + s = 0)
  (h2 : q * k^3 + r * k^2 + s * k + p = 0)
  (h3 : 3 * p * k^2 + 2 * q * k + r = 0) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_root_values_l2857_285728


namespace NUMINAMATH_CALUDE_plant_arrangement_count_l2857_285701

/-- Represents the number of basil plants -/
def num_basil : ℕ := 4

/-- Represents the number of tomato plants -/
def num_tomato : ℕ := 4

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem stating the number of ways to arrange the plants -/
theorem plant_arrangement_count : 
  factorial (num_basil + 1) * factorial num_tomato = 2880 := by
  sorry

end NUMINAMATH_CALUDE_plant_arrangement_count_l2857_285701


namespace NUMINAMATH_CALUDE_multiple_birth_statistics_l2857_285714

theorem multiple_birth_statistics (total_babies : ℕ) 
  (h_total : total_babies = 1200) 
  (twins triplets quintuplets : ℕ) 
  (h_twins : twins = 3 * triplets) 
  (h_triplets : triplets = 2 * quintuplets) 
  (h_sum : 2 * twins + 3 * triplets + 5 * quintuplets = total_babies) : 
  5 * quintuplets = 260 := by
  sorry

end NUMINAMATH_CALUDE_multiple_birth_statistics_l2857_285714


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l2857_285777

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

theorem base_conversion_subtraction :
  let base_5_num := [1, 3, 4, 2, 5]
  let base_8_num := [2, 3, 4, 1]
  to_base_10 base_5_num 5 - to_base_10 base_8_num 8 = 2697 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l2857_285777


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_of_10_15_25_l2857_285717

theorem gcf_lcm_sum_of_10_15_25 : ∃ (A B : ℕ),
  (A = Nat.gcd 10 (Nat.gcd 15 25)) ∧
  (B = Nat.lcm 10 (Nat.lcm 15 25)) ∧
  (A + B = 155) := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_of_10_15_25_l2857_285717


namespace NUMINAMATH_CALUDE_probability_two_defective_in_four_tests_l2857_285709

def total_components : ℕ := 6
def defective_components : ℕ := 2
def good_components : ℕ := 4
def tests : ℕ := 4

theorem probability_two_defective_in_four_tests :
  (
    -- Probability of finding one defective in first three tests and second on fourth test
    (defective_components / total_components *
     good_components / (total_components - 1) *
     (good_components - 1) / (total_components - 2) *
     (defective_components - 1) / (total_components - 3)) +
    (good_components / total_components *
     defective_components / (total_components - 1) *
     (good_components - 1) / (total_components - 2) *
     (defective_components - 1) / (total_components - 3)) +
    (good_components / total_components *
     (good_components - 1) / (total_components - 1) *
     defective_components / (total_components - 2) *
     (defective_components - 1) / (total_components - 3)) +
    -- Probability of finding all good components in four tests
    (good_components / total_components *
     (good_components - 1) / (total_components - 1) *
     (good_components - 2) / (total_components - 2) *
     (good_components - 3) / (total_components - 3))
  ) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_defective_in_four_tests_l2857_285709


namespace NUMINAMATH_CALUDE_max_balances_correct_max_balances_achievable_l2857_285749

/-- Represents a set of unique weights -/
def UniqueWeights (n : ℕ) := Fin n → ℝ

/-- Represents the state of the balance scale -/
structure BalanceState where
  left : List ℝ
  right : List ℝ

/-- Checks if the balance scale is in equilibrium -/
def isBalanced (state : BalanceState) : Prop :=
  state.left.sum = state.right.sum

/-- Represents a sequence of weight placements -/
def WeightPlacement (n : ℕ) := Fin n → Bool × Fin n

/-- Counts the number of times the scale balances during a sequence of weight placements -/
def countBalances (weights : UniqueWeights 2021) (placements : WeightPlacement m) : ℕ :=
  sorry

/-- The maximum number of times the scale can balance -/
def maxBalances : ℕ := 673

theorem max_balances_correct (weights : UniqueWeights 2021) :
  ∀ (m : ℕ) (placements : WeightPlacement m),
    countBalances weights placements ≤ maxBalances :=
  sorry

theorem max_balances_achievable :
  ∃ (weights : UniqueWeights 2021) (m : ℕ) (placements : WeightPlacement m),
    countBalances weights placements = maxBalances :=
  sorry

end NUMINAMATH_CALUDE_max_balances_correct_max_balances_achievable_l2857_285749


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l2857_285721

/-- A right parallelepiped with a parallelogram base -/
structure RightParallelepiped where
  base_angle : ℝ
  base_area : ℝ
  lateral_face_area1 : ℝ
  lateral_face_area2 : ℝ

/-- The volume of the right parallelepiped -/
def volume (p : RightParallelepiped) : ℝ := sorry

theorem parallelepiped_volume (p : RightParallelepiped) 
  (h1 : p.base_angle = π / 6)
  (h2 : p.base_area = 4)
  (h3 : p.lateral_face_area1 = 6)
  (h4 : p.lateral_face_area2 = 12) :
  volume p = 12 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l2857_285721


namespace NUMINAMATH_CALUDE_road_trip_distance_l2857_285780

theorem road_trip_distance (D : ℝ) 
  (h1 : D > 0)
  (h2 : D * (1 - 1/3) * (1 - 1/5) * (1 - 1/4) = 400) : D = 1000 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_distance_l2857_285780


namespace NUMINAMATH_CALUDE_unique_numbers_satisfying_conditions_l2857_285743

theorem unique_numbers_satisfying_conditions : 
  ∃! (x y : ℕ), 
    10 ≤ x ∧ x < 100 ∧
    100 ≤ y ∧ y < 1000 ∧
    1000 * x + y = 8 * x * y ∧
    x + y = 141 := by sorry

end NUMINAMATH_CALUDE_unique_numbers_satisfying_conditions_l2857_285743


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l2857_285735

theorem lcm_from_product_and_hcf (a b : ℕ+) :
  a * b = 45276 →
  Nat.gcd a b = 22 →
  Nat.lcm a b = 2058 := by
sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l2857_285735


namespace NUMINAMATH_CALUDE_pastry_eating_time_l2857_285715

/-- The time it takes for two people to eat a certain number of pastries together -/
def eating_time (quick_rate : ℚ) (slow_rate : ℚ) (total_pastries : ℚ) : ℚ :=
  total_pastries / (quick_rate + slow_rate)

/-- Theorem stating the time it takes Miss Quick and Miss Slow to eat 5 pastries together -/
theorem pastry_eating_time :
  let quick_rate : ℚ := 1 / 15
  let slow_rate : ℚ := 1 / 25
  let total_pastries : ℚ := 5
  eating_time quick_rate slow_rate total_pastries = 375 / 8 := by
sorry

end NUMINAMATH_CALUDE_pastry_eating_time_l2857_285715


namespace NUMINAMATH_CALUDE_complex_polynomial_root_l2857_285757

theorem complex_polynomial_root (a b c : ℤ) : 
  (a * (1 + Complex.I * Real.sqrt 3)^3 + b * (1 + Complex.I * Real.sqrt 3)^2 + c * (1 + Complex.I * Real.sqrt 3) + b + a = 0) →
  (Int.gcd a (Int.gcd b c) = 1) →
  (abs c = 9) := by
  sorry

end NUMINAMATH_CALUDE_complex_polynomial_root_l2857_285757


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l2857_285791

theorem exam_maximum_marks :
  ∀ (max_marks : ℕ) (student_marks : ℕ) (failing_margin : ℕ),
    student_marks = 92 →
    failing_margin = 40 →
    (student_marks + failing_margin : ℚ) = (33 / 100) * max_marks →
    max_marks = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l2857_285791


namespace NUMINAMATH_CALUDE_martin_additional_hens_l2857_285703

/-- Represents the farm's hen and egg production scenario --/
structure FarmScenario where
  initial_hens : ℕ
  initial_days : ℕ
  initial_eggs : ℕ
  final_days : ℕ
  final_eggs : ℕ

/-- Calculates the number of additional hens needed --/
def additional_hens_needed (scenario : FarmScenario) : ℕ :=
  let eggs_per_hen := scenario.final_eggs * scenario.initial_days / (scenario.final_days * scenario.initial_eggs)
  let total_hens_needed := scenario.final_eggs / (eggs_per_hen * scenario.final_days / scenario.initial_days)
  total_hens_needed - scenario.initial_hens

/-- The main theorem stating the number of additional hens Martin needs to buy --/
theorem martin_additional_hens :
  let scenario : FarmScenario := {
    initial_hens := 10,
    initial_days := 10,
    initial_eggs := 80,
    final_days := 15,
    final_eggs := 300
  }
  additional_hens_needed scenario = 15 := by
  sorry

end NUMINAMATH_CALUDE_martin_additional_hens_l2857_285703


namespace NUMINAMATH_CALUDE_min_shift_for_odd_cosine_l2857_285747

/-- Given a function f(x) = cos(2x + π/6) that is shifted right by φ units,
    prove that the minimum positive φ that makes the resulting function odd is π/3. -/
theorem min_shift_for_odd_cosine :
  let f (x : ℝ) := Real.cos (2 * x + π / 6)
  let g (φ : ℝ) (x : ℝ) := f (x - φ)
  ∀ φ : ℝ, φ > 0 →
    (∀ x : ℝ, g φ (-x) = -(g φ x)) →
    φ ≥ π / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_shift_for_odd_cosine_l2857_285747


namespace NUMINAMATH_CALUDE_q_duration_is_nine_l2857_285785

/-- Investment and profit ratios for partners P and Q -/
structure PartnershipRatios where
  investment_ratio_p : ℕ
  investment_ratio_q : ℕ
  profit_ratio_p : ℕ
  profit_ratio_q : ℕ

/-- Calculate the investment duration of partner Q given the ratios and P's duration -/
def calculate_q_duration (ratios : PartnershipRatios) (p_duration : ℕ) : ℕ :=
  (ratios.investment_ratio_p * p_duration * ratios.profit_ratio_q) / 
  (ratios.investment_ratio_q * ratios.profit_ratio_p)

/-- Theorem stating that Q's investment duration is 9 months given the specified ratios and P's duration -/
theorem q_duration_is_nine :
  let ratios : PartnershipRatios := {
    investment_ratio_p := 7,
    investment_ratio_q := 5,
    profit_ratio_p := 7,
    profit_ratio_q := 9
  }
  let p_duration := 5
  calculate_q_duration ratios p_duration = 9 := by
  sorry

end NUMINAMATH_CALUDE_q_duration_is_nine_l2857_285785


namespace NUMINAMATH_CALUDE_time_to_write_rearrangements_l2857_285765

/-- The time required to write all rearrangements of a name -/
theorem time_to_write_rearrangements 
  (num_letters : ℕ) 
  (rearrangements_per_minute : ℕ) 
  (h1 : num_letters = 5) 
  (h2 : rearrangements_per_minute = 15) : 
  (Nat.factorial num_letters : ℚ) / (rearrangements_per_minute * 60 : ℚ) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_time_to_write_rearrangements_l2857_285765


namespace NUMINAMATH_CALUDE_marias_workday_ends_at_5pm_l2857_285711

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  deriving Repr

/-- Represents a workday -/
structure Workday where
  start : Time
  totalWorkHours : Nat
  lunchBreakStart : Time
  lunchBreakDuration : Nat
  deriving Repr

def addHours (t : Time) (h : Nat) : Time :=
  { hour := (t.hour + h) % 24, minute := t.minute }

def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute + m
  { hour := totalMinutes / 60, minute := totalMinutes % 60 }

def mariasWorkday : Workday :=
  { start := { hour := 8, minute := 0 },
    totalWorkHours := 8,
    lunchBreakStart := { hour := 13, minute := 0 },
    lunchBreakDuration := 1 }

theorem marias_workday_ends_at_5pm :
  let endTime := addHours (addMinutes mariasWorkday.lunchBreakStart mariasWorkday.lunchBreakDuration)
                          (mariasWorkday.totalWorkHours - (mariasWorkday.lunchBreakStart.hour - mariasWorkday.start.hour))
  endTime = { hour := 17, minute := 0 } :=
by sorry

end NUMINAMATH_CALUDE_marias_workday_ends_at_5pm_l2857_285711


namespace NUMINAMATH_CALUDE_max_sequence_length_sequence_of_length_12_exists_l2857_285776

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The property that the sum of any five consecutive terms is negative -/
def SumOfFiveNegative (a : Sequence) :=
  ∀ i, i > 0 → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4)) < 0

/-- The property that the sum of any nine consecutive terms is positive -/
def SumOfNinePositive (a : Sequence) :=
  ∀ i, i > 0 → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) + a (i+7) + a (i+8)) > 0

/-- The maximum length of a sequence satisfying both properties is 12 -/
theorem max_sequence_length :
  ∀ n : ℕ, n > 12 →
    ¬∃ a : Sequence, (SumOfFiveNegative a ∧ SumOfNinePositive a ∧ ∀ i > n, a i = 0) :=
by sorry

/-- There exists a sequence of length 12 satisfying both properties -/
theorem sequence_of_length_12_exists :
  ∃ a : Sequence, SumOfFiveNegative a ∧ SumOfNinePositive a ∧ ∀ i > 12, a i = 0 :=
by sorry

end NUMINAMATH_CALUDE_max_sequence_length_sequence_of_length_12_exists_l2857_285776


namespace NUMINAMATH_CALUDE_exact_one_root_at_most_one_root_l2857_285745

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : Prop := a * x^2 + 2*x + 1 = 0

-- Define the set of roots
def root_set (a : ℝ) : Set ℝ := {x | quadratic_equation a x}

-- Statement 1: A contains exactly one element iff a = 1 or a = 0
theorem exact_one_root (a : ℝ) : 
  (∃! x, x ∈ root_set a) ↔ (a = 1 ∨ a = 0) :=
sorry

-- Statement 2: A contains at most one element iff a ∈ {0} ∪ [1, +∞)
theorem at_most_one_root (a : ℝ) :
  (∀ x y, x ∈ root_set a → y ∈ root_set a → x = y) ↔ (a = 0 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_exact_one_root_at_most_one_root_l2857_285745


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2857_285740

theorem polynomial_coefficient_sum : ∀ P Q R S : ℝ,
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = P * x^3 + Q * x^2 + R * x + S) →
  P + Q + R + S = 36 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2857_285740


namespace NUMINAMATH_CALUDE_absolute_value_expression_l2857_285710

theorem absolute_value_expression (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c > 0) : 
  |a| - |a + b| + |c - a| + |b - c| = 2 * c - a := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l2857_285710


namespace NUMINAMATH_CALUDE_alien_energy_conversion_l2857_285779

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the alien's energy units --/
def alienEnergy : List Nat := [3, 6, 2]

theorem alien_energy_conversion :
  base7ToBase10 alienEnergy = 143 := by
  sorry

end NUMINAMATH_CALUDE_alien_energy_conversion_l2857_285779


namespace NUMINAMATH_CALUDE_cubic_root_proof_l2857_285799

theorem cubic_root_proof :
  let x : ℝ := (Real.rpow 81 (1/3) + Real.rpow 9 (1/3) + 1) / 27
  27 * x^3 - 6 * x^2 - 6 * x - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_proof_l2857_285799


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_4_l2857_285734

/-- A geometric sequence with its sum -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1  -- Geometric sequence property

/-- Theorem stating the conditions and the result to be proved -/
theorem geometric_sequence_sum_4 (seq : GeometricSequence) 
  (h1 : seq.a 1 - seq.a 2 = 2)
  (h2 : seq.a 2 - seq.a 3 = 6) :
  seq.S 4 = -40 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_4_l2857_285734


namespace NUMINAMATH_CALUDE_train_trip_time_difference_l2857_285733

/-- The time difference between two trips with given distance and speeds -/
theorem train_trip_time_difference 
  (distance : ℝ) 
  (speed_outbound speed_return : ℝ) 
  (h1 : distance = 480) 
  (h2 : speed_outbound = 160) 
  (h3 : speed_return = 120) : 
  distance / speed_return - distance / speed_outbound = 1 := by
sorry

end NUMINAMATH_CALUDE_train_trip_time_difference_l2857_285733


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l2857_285719

/-- A right trapezoid with specific properties -/
structure RightTrapezoid where
  /-- The length of one lateral side -/
  side1 : ℝ
  /-- The length of the other lateral side -/
  side2 : ℝ
  /-- The diagonal bisects the acute angle -/
  diagonal_bisects_acute_angle : Bool

/-- The area of the right trapezoid -/
def area (t : RightTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific right trapezoid is 104 -/
theorem specific_trapezoid_area :
  ∀ (t : RightTrapezoid),
    t.side1 = 10 ∧
    t.side2 = 8 ∧
    t.diagonal_bisects_acute_angle = true →
    area t = 104 :=
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l2857_285719


namespace NUMINAMATH_CALUDE_parallel_vectors_m_l2857_285770

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m (a b : ℝ × ℝ) (h : vector_parallel a b) :
  a = (2, -1) → b = (-1, 1/2) := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_l2857_285770


namespace NUMINAMATH_CALUDE_total_income_scientific_notation_exponent_l2857_285754

/-- Represents the average annual income from 1 acre of medicinal herbs in dollars -/
def average_income_per_acre : ℝ := 20000

/-- Represents the number of acres of medicinal herbs planted in the county -/
def acres_planted : ℝ := 8000

/-- Calculates the total annual income from medicinal herbs in the county -/
def total_income : ℝ := average_income_per_acre * acres_planted

/-- Represents the exponent in the scientific notation of the total income -/
def n : ℕ := 8

/-- Theorem stating that the exponent in the scientific notation of the total income is 8 -/
theorem total_income_scientific_notation_exponent : 
  ∃ (a : ℝ), a > 1 ∧ a < 10 ∧ total_income = a * (10 : ℝ) ^ n :=
sorry

end NUMINAMATH_CALUDE_total_income_scientific_notation_exponent_l2857_285754


namespace NUMINAMATH_CALUDE_commission_per_car_l2857_285778

/-- Proves that the commission per car is $200 given the specified conditions -/
theorem commission_per_car 
  (base_salary : ℕ) 
  (march_earnings : ℕ) 
  (cars_to_double : ℕ) 
  (h1 : base_salary = 1000)
  (h2 : march_earnings = 2000)
  (h3 : cars_to_double = 15) :
  (2 * march_earnings - base_salary) / cars_to_double = 200 := by
  sorry

end NUMINAMATH_CALUDE_commission_per_car_l2857_285778


namespace NUMINAMATH_CALUDE_fruit_basket_ratio_l2857_285723

/-- Fruit basket problem -/
theorem fruit_basket_ratio : 
  ∀ (oranges apples bananas peaches : ℕ),
  oranges = 6 →
  apples = oranges - 2 →
  bananas = 3 * apples →
  oranges + apples + bananas + peaches = 28 →
  peaches * 2 = bananas :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_ratio_l2857_285723


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2857_285724

theorem trigonometric_simplification :
  let f : ℝ → ℝ := λ x => Real.sin (x * π / 180)
  let g : ℝ → ℝ := λ x => Real.cos (x * π / 180)
  (f 15 + f 25 + f 35 + f 45 + f 55 + f 65 + f 75 + f 85) / (g 10 * g 15 * g 30) =
  8 * Real.sqrt 3 * g 40 * g 5 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2857_285724


namespace NUMINAMATH_CALUDE_pyramid_volume_l2857_285746

/-- Represents a pyramid with a triangular base --/
structure TriangularPyramid where
  base_side1 : ℝ
  base_side2 : ℝ
  base_side3 : ℝ
  lateral_angle : ℝ

/-- Calculates the volume of a triangular pyramid --/
def volume (p : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating that a pyramid with the given properties has a volume of 6 --/
theorem pyramid_volume :
  ∀ (p : TriangularPyramid),
    p.base_side1 = 6 ∧
    p.base_side2 = 5 ∧
    p.base_side3 = 5 ∧
    p.lateral_angle = π / 4 →
    volume p = 6 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l2857_285746


namespace NUMINAMATH_CALUDE_outfit_count_l2857_285706

/-- Represents the number of shirts available -/
def num_shirts : ℕ := 7

/-- Represents the number of pants available -/
def num_pants : ℕ := 5

/-- Represents the number of ties available -/
def num_ties : ℕ := 4

/-- Represents the total number of tie options (including the option of not wearing a tie) -/
def tie_options : ℕ := num_ties + 1

/-- Calculates the total number of possible outfits -/
def total_outfits : ℕ := num_shirts * num_pants * tie_options

/-- Theorem stating that the total number of possible outfits is 175 -/
theorem outfit_count : total_outfits = 175 := by sorry

end NUMINAMATH_CALUDE_outfit_count_l2857_285706


namespace NUMINAMATH_CALUDE_sin_cos_equation_solutions_l2857_285793

theorem sin_cos_equation_solutions (π : Real) (sin cos : Real → Real) :
  (∃ (x₁ x₂ : Real), x₁ ≠ x₂ ∧ 
   0 ≤ x₁ ∧ x₁ ≤ π ∧ 
   0 ≤ x₂ ∧ x₂ ≤ π ∧
   sin (π / 2 * cos x₁) = cos (π / 2 * sin x₁) ∧
   sin (π / 2 * cos x₂) = cos (π / 2 * sin x₂)) ∧
  (∀ (x y z : Real), 
   0 ≤ x ∧ x ≤ π ∧ 
   0 ≤ y ∧ y ≤ π ∧ 
   0 ≤ z ∧ z ≤ π ∧
   x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
   sin (π / 2 * cos x) = cos (π / 2 * sin x) ∧
   sin (π / 2 * cos y) = cos (π / 2 * sin y) ∧
   sin (π / 2 * cos z) = cos (π / 2 * sin z) →
   False) :=
by sorry


end NUMINAMATH_CALUDE_sin_cos_equation_solutions_l2857_285793


namespace NUMINAMATH_CALUDE_julio_earnings_l2857_285729

/-- Julio's earnings calculation --/
theorem julio_earnings (commission_per_customer : ℕ) (first_week_customers : ℕ) 
  (salary : ℕ) (bonus : ℕ) : 
  commission_per_customer = 1 →
  first_week_customers = 35 →
  salary = 500 →
  bonus = 50 →
  (commission_per_customer * (first_week_customers + 2 * first_week_customers + 3 * first_week_customers) + 
   salary + bonus) = 760 := by
  sorry

#check julio_earnings

end NUMINAMATH_CALUDE_julio_earnings_l2857_285729


namespace NUMINAMATH_CALUDE_abs_plus_one_positive_l2857_285752

theorem abs_plus_one_positive (a : ℚ) : 0 < |a| + 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_plus_one_positive_l2857_285752


namespace NUMINAMATH_CALUDE_sum_equals_210_l2857_285762

theorem sum_equals_210 : 145 + 35 + 25 + 5 = 210 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_210_l2857_285762


namespace NUMINAMATH_CALUDE_unique_fraction_decomposition_l2857_285738

theorem unique_fraction_decomposition (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (m n : ℕ), m ≠ n ∧ 2 / p = 1 / n + 1 / m ∧
  n = (p + 1) / 2 ∧ m = p * (p + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_fraction_decomposition_l2857_285738


namespace NUMINAMATH_CALUDE_common_chord_length_is_2sqrt5_l2857_285789

/-- Circle C1 with equation x^2 + y^2 + 2x + 8y - 8 = 0 -/
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0

/-- Circle C2 with equation x^2 + y^2 - 4x - 4y - 2 = 0 -/
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

/-- The circles C1 and C2 intersect -/
axiom circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y

/-- The length of the common chord of two intersecting circles -/
def common_chord_length (C1 C2 : ℝ → ℝ → Prop) : ℝ := sorry

/-- Theorem: The length of the common chord of C1 and C2 is 2√5 -/
theorem common_chord_length_is_2sqrt5 :
  common_chord_length C1 C2 = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_common_chord_length_is_2sqrt5_l2857_285789


namespace NUMINAMATH_CALUDE_rectangle_with_tangent_circle_l2857_285707

theorem rectangle_with_tangent_circle 
  (r : ℝ) 
  (h1 : r = 6) 
  (A_circle : ℝ) 
  (h2 : A_circle = π * r^2) 
  (A_rectangle : ℝ) 
  (h3 : A_rectangle = 3 * A_circle) 
  (shorter_side : ℝ) 
  (h4 : shorter_side = 2 * r) 
  (longer_side : ℝ) 
  (h5 : A_rectangle = shorter_side * longer_side) : 
  longer_side = 9 * π := by
sorry

end NUMINAMATH_CALUDE_rectangle_with_tangent_circle_l2857_285707


namespace NUMINAMATH_CALUDE_square_sum_difference_l2857_285725

theorem square_sum_difference (a b : ℝ) 
  (h1 : (a + b)^2 = 17) 
  (h2 : (a - b)^2 = 11) : 
  a^2 + b^2 = 14 := by sorry

end NUMINAMATH_CALUDE_square_sum_difference_l2857_285725


namespace NUMINAMATH_CALUDE_dividend_percentage_calculation_l2857_285774

/-- Calculates the dividend percentage given investment details and dividend received -/
theorem dividend_percentage_calculation
  (investment : ℝ)
  (share_face_value : ℝ)
  (premium_percentage : ℝ)
  (dividend_received : ℝ)
  (h1 : investment = 14400)
  (h2 : share_face_value = 100)
  (h3 : premium_percentage = 20)
  (h4 : dividend_received = 720) :
  let share_cost := share_face_value * (1 + premium_percentage / 100)
  let num_shares := investment / share_cost
  let dividend_per_share := dividend_received / num_shares
  let dividend_percentage := (dividend_per_share / share_face_value) * 100
  dividend_percentage = 6 := by
sorry

end NUMINAMATH_CALUDE_dividend_percentage_calculation_l2857_285774


namespace NUMINAMATH_CALUDE_factors_of_M_l2857_285786

/-- The number of natural-number factors of M, where M = 2^2 · 3^3 · 5^2 · 7^1 -/
def number_of_factors (M : ℕ) : ℕ :=
  if M = 2^2 * 3^3 * 5^2 * 7^1 then 72 else 0

/-- Theorem stating that the number of natural-number factors of M is 72 -/
theorem factors_of_M :
  number_of_factors (2^2 * 3^3 * 5^2 * 7^1) = 72 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_M_l2857_285786


namespace NUMINAMATH_CALUDE_greatest_difference_units_digit_l2857_285769

theorem greatest_difference_units_digit (x : ℕ) :
  x < 10 →
  (840 + x) % 3 = 0 →
  ∃ y, y < 10 ∧ (840 + y) % 3 = 0 ∧ 
  ∀ z, z < 10 → (840 + z) % 3 = 0 → (max x y - min x y) ≥ (max x z - min x z) :=
by sorry

end NUMINAMATH_CALUDE_greatest_difference_units_digit_l2857_285769


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l2857_285742

def q (x : ℝ) : ℝ := -10 * x^2 + 40 * x - 30

def numerator (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 4 * x + 6

theorem q_satisfies_conditions :
  (∀ x, x ≠ 1 ∧ x ≠ 3 → q x ≠ 0) ∧
  (q 1 = 0 ∧ q 3 = 0) ∧
  (∀ x, x ≠ 1 ∧ x ≠ 3 → ∃ y, y = numerator x / q x) ∧
  (¬ ∃ L, ∀ ε > 0, ∃ δ > 0, ∀ x, |x| > δ → |numerator x / q x - L| < ε) ∧
  q 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l2857_285742


namespace NUMINAMATH_CALUDE_no_real_b_for_single_solution_l2857_285783

-- Define the quadratic function g(x) with parameter b
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + 3*b*x + 4*b

-- Theorem stating that no real b exists such that g(x) has its vertex at y = 5
theorem no_real_b_for_single_solution :
  ¬ ∃ b : ℝ, ∃ x : ℝ, g b x = 5 ∧ ∀ y : ℝ, g b y ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_no_real_b_for_single_solution_l2857_285783
