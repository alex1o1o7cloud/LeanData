import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l1347_134741

/-- Represents a cubic polynomial of the form 3x³ + dx² + ex + 6 -/
structure CubicPolynomial where
  d : ℝ
  e : ℝ

/-- The sum of the zeros of a cubic polynomial -/
noncomputable def sum_of_zeros (p : CubicPolynomial) : ℝ := -p.d / 3

/-- The product of the zeros of a cubic polynomial -/
def product_of_zeros : ℝ := -2

/-- The mean of the zeros of a cubic polynomial -/
noncomputable def mean_of_zeros (p : CubicPolynomial) : ℝ := sum_of_zeros p / 3

/-- The sum of the coefficients of the cubic polynomial -/
def sum_of_coefficients (p : CubicPolynomial) : ℝ := 3 + p.d + p.e + 6

theorem cubic_polynomial_property (p : CubicPolynomial) :
  mean_of_zeros p = 2 * product_of_zeros →
  mean_of_zeros p = sum_of_coefficients p →
  p.e = -49 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l1347_134741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vectors_theorem_l1347_134765

-- Define the direction vectors
def a (x : ℝ) : Fin 3 → ℝ := ![2, 4, x]
def b (y : ℝ) : Fin 3 → ℝ := ![2, y, 2]

-- State the theorem
theorem direction_vectors_theorem (x y : ℝ) : 
  (Real.sqrt ((a x 0)^2 + (a x 1)^2 + (a x 2)^2) = 6) → 
  ((a x 0) * (b y 0) + (a x 1) * (b y 1) + (a x 2) * (b y 2) = 0) → 
  (x + y = -3 ∨ x + y = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vectors_theorem_l1347_134765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_range_of_a_when_b_is_one_l1347_134730

-- Define the function f as noncomputable
noncomputable def f (a b x : ℝ) : ℝ := (Real.exp x - 1) / x - a * x - b

-- Part 1: Prove that a = 3/2 and b = 2 given the tangent line condition
theorem tangent_line_condition (a b : ℝ) :
  (∀ x y : ℝ, y = f a b x → x + 2 * y + 4 = 0 → x = 1) →
  a = 3/2 ∧ b = 2 := by
  sorry

-- Part 2: Prove the range of a when b = 1
theorem range_of_a_when_b_is_one (a : ℝ) :
  (∃ m : ℝ, m < 0 ∧ ∀ x ∈ Set.Ioo m 0, f a 1 x < 0) ↔
  ∃ m : ℝ, m < 0 ∧ a ≤ (Real.exp m - 1 - m) / (m^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_range_of_a_when_b_is_one_l1347_134730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_mode_median_l1347_134755

/-- Helper function to calculate the mode of a list -/
noncomputable def mode (l : List ℝ) : ℝ := sorry

/-- Helper function to calculate the median of a list -/
noncomputable def median (l : List ℝ) : ℝ := sorry

/-- Given two sets of data with specified properties, prove the mode and median of the combined set -/
theorem data_set_mode_median :
  ∀ (a b : ℝ),
  (3 + a + 2*b + 5) / 4 = 6 →
  (a + 6 + b) / 3 = 6 →
  let combined_set := [3, a, 2*b, 5, a, 6, b]
  (mode combined_set = 8 ∧ median combined_set = 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_mode_median_l1347_134755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1347_134772

noncomputable def f (x : ℝ) := Real.log (x^2 - 2*x - 8)

theorem f_monotone_increasing : 
  StrictMonoOn f (Set.Ioi 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1347_134772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_four_is_twelve_l1347_134705

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  second_term : a 2 = 4
  special_relation : a 1 + a 5 = 4 * a 3 - 4

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

/-- Theorem: The sum of the first 4 terms of the given arithmetic sequence is 12 -/
theorem sum_four_is_twelve (seq : ArithmeticSequence) : sum_n seq 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_four_is_twelve_l1347_134705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_properties_l1347_134770

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := 2 * (Real.cos ((1/3) * x + Real.pi/4))^2 - 1
def g (x : ℝ) : ℝ := Real.sin (2*x + 5*Real.pi/4)
def h (x : ℝ) : ℝ := Real.sin (2*x + Real.pi/3)

-- State the theorem
theorem trigonometric_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ α, Real.sin α + Real.cos α ≤ Real.sqrt 2) ∧
  (∀ α β, 0 < α ∧ α < β ∧ β < Real.pi/2 → Real.tan α < Real.tan β) ∧
  (∀ x, g (Real.pi/4 - x) = g (Real.pi/4 + x)) ∧
  (∃ x, h (Real.pi/6 - x) ≠ h (Real.pi/6 + x)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_properties_l1347_134770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_replacement_l1347_134742

/-- Represents a sugar solution with a given weight and sugar percentage -/
structure SugarSolution where
  weight : ℝ
  sugarPercentage : ℝ

/-- Calculates the amount of sugar in a solution -/
noncomputable def sugarAmount (solution : SugarSolution) : ℝ :=
  solution.weight * (solution.sugarPercentage / 100)

/-- Theorem: If a 10% sugar solution has 1/4 replaced by another solution
    resulting in an 18% sugar solution, the replacing solution is 42% sugar -/
theorem sugar_solution_replacement
  (original : SugarSolution)
  (replaced : SugarSolution)
  (result : SugarSolution)
  (h1 : original.sugarPercentage = 10)
  (h2 : replaced.weight = original.weight / 4)
  (h3 : result.weight = original.weight)
  (h4 : result.sugarPercentage = 18)
  (h5 : sugarAmount result = sugarAmount original - sugarAmount original / 4 + sugarAmount replaced) :
  replaced.sugarPercentage = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_replacement_l1347_134742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paris_hair_pigeonhole_l1347_134704

/-- The number of hairs on a person's head -/
def hair_count : ℕ → ℕ := sorry

/-- Theorem: There exist at least 4 Parisians with the same number of hairs -/
theorem paris_hair_pigeonhole (population : ℕ) (max_hairs : ℕ) :
  population = 2000000 →
  max_hairs = 600000 →
  ∃ (n : ℕ), n ≤ max_hairs ∧ 
    (∃ (group : Finset ℕ), group.card ≥ 4 ∧ 
      ∀ i ∈ group, i < population ∧ hair_count i = n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paris_hair_pigeonhole_l1347_134704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_one_l1347_134779

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function y = a * 3^x + 1 / 3^x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (3 : ℝ) ^ x + (3 : ℝ) ^ (-x)

theorem even_function_implies_a_equals_one :
  ∀ a : ℝ, IsEven (f a) → a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_one_l1347_134779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_repeat_b_l1347_134769

noncomputable def b : ℕ → ℝ
  | 0 => Real.cos (Real.pi / 18) ^ 2
  | n + 1 => 4 * b n * (1 - b n)

theorem smallest_repeat_b : (∃ n : ℕ, n > 0 ∧ b n = b 0) ∧ (∀ m : ℕ, 0 < m → m < 30 → b m ≠ b 0) ∧ b 30 = b 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_repeat_b_l1347_134769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_is_three_l1347_134760

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := 8 * (Real.cos t) ^ 3
noncomputable def y (t : ℝ) : ℝ := 8 * (Real.sin t) ^ 3

-- Define the arc length function
noncomputable def arcLength (a b : ℝ) : ℝ :=
  ∫ t in a..b, Real.sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2)

-- State the theorem
theorem arc_length_is_three :
  arcLength 0 (Real.pi / 6) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_is_three_l1347_134760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_cos6_l1347_134768

theorem min_sin6_cos6 :
  ∀ x : ℝ, Real.sin x ^ 6 + Real.cos x ^ 6 ≥ 3 / 4 :=
by
  intro x
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_cos6_l1347_134768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1347_134764

theorem problem_statement (m : ℝ) (a b : ℝ) 
  (h1 : (9 : ℝ)^m = 10)
  (h2 : a = (10 : ℝ)^m - 11)
  (h3 : b = (8 : ℝ)^m - 9) :
  a > 0 ∧ 0 > b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1347_134764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_to_inclination_angle_l1347_134792

noncomputable def Line2D := ℝ × ℝ → Prop

structure LineProperties (l : Line2D) :=
  (slope : ℝ)
  (inclinationAngle : ℝ)

theorem line_slope_to_inclination_angle (l : Line2D) (props : LineProperties l) :
  props.slope = -Real.sqrt 3 / 3 → props.inclinationAngle = 150 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_to_inclination_angle_l1347_134792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_45_24567_to_nearest_tenth_l1347_134709

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The theorem stating that rounding 45.24567 to the nearest tenth equals 45.2 -/
theorem round_45_24567_to_nearest_tenth :
  roundToNearestTenth 45.24567 = 45.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_45_24567_to_nearest_tenth_l1347_134709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1347_134732

-- Define the hyperbola and circle
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def myCircle (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 15 = 0

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt ((a^2 + b^2) / a^2)

-- State the theorem
theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_intersect : ∃ (x y : ℝ), hyperbola a b x y ∧ myCircle x y ∧ 
    ∃ (x' y' : ℝ), hyperbola a b x' y' ∧ myCircle x' y' ∧ 
    (x - x')^2 + (y - y')^2 = 2) :
  eccentricity a b = 4 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1347_134732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_offerings_for_seniors_l1347_134758

/-- A simplified model of bank offerings for seniors --/
theorem bank_offerings_for_seniors (deposit_rate loan_rate : ℝ) :
  deposit_rate > 0 ∧ loan_rate > 0 →
  ∃ (senior_deposit_rate senior_loan_rate : ℝ),
    senior_deposit_rate > deposit_rate ∧
    senior_loan_rate < loan_rate :=
by
  intro h
  -- The actual proof would go here, but we'll use sorry for now
  sorry

#check bank_offerings_for_seniors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_offerings_for_seniors_l1347_134758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l1347_134797

/-- The central angle that maximizes the area of a sector with given perimeter -/
noncomputable def max_area_angle (a : ℝ) : ℝ := 2

/-- The maximum area of a sector with given perimeter -/
noncomputable def max_area (a : ℝ) : ℝ := a^2 / 4

/-- Theorem stating the conditions for maximum area of a sector -/
theorem sector_max_area (a : ℝ) (h : a > 0) :
  let r := a / 4
  let l := a / 2
  let α := max_area_angle a
  let S := max_area a
  (∀ (r' : ℝ), 0 < r' ∧ r' < a / 2 →
    r' * (a - 2 * r') / 2 ≤ S) ∧
  α = l / r ∧
  S = r * l / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l1347_134797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rahul_batting_average_l1347_134747

theorem rahul_batting_average (current_avg : ℝ) (new_avg : ℝ) (new_runs : ℝ) :
  current_avg = 46 →
  new_avg = 54 →
  new_runs = 78 →
  (∃ m : ℕ, m * current_avg + new_runs = (m + 1) * new_avg ∧ m = 3) :=
by
  intros h_current h_new h_runs
  use 3
  constructor
  · simp [h_current, h_new, h_runs]
    norm_num
  · rfl

#check rahul_batting_average

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rahul_batting_average_l1347_134747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_max_value_l1347_134715

theorem geometric_mean_max_value (a b : ℝ) : 
  a^2 = (1 + 2*b) * (1 - 2*b) → 
  (∀ x y : ℝ, x^2 = (1 + 2*y) * (1 - 2*y) → 
    2*a*b / (|a| + 2*|b|) ≤ 2*x*y / (|x| + 2*|y|)) →
  2*a*b / (|a| + 2*|b|) = Real.sqrt 2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_max_value_l1347_134715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trout_ratio_l1347_134720

/-- The ratio of trouts caught by Tom to Melanie is 2:1 -/
theorem trout_ratio : 
  (16 : ℚ) / 8 = 2 := by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trout_ratio_l1347_134720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_expenditure_feb_to_jul_l1347_134721

/-- Calculate the average expenditure for February to July given the following conditions:
  * The average expenditure for January to June is 4200
  * The expenditure in January is 1200
  * The expenditure in July is 1500
-/
theorem average_expenditure_feb_to_jul 
  (avg_jan_to_jun : ℝ) 
  (exp_jan : ℝ) 
  (exp_jul : ℝ) 
  (h1 : avg_jan_to_jun = 4200) 
  (h2 : exp_jan = 1200) 
  (h3 : exp_jul = 1500) : 
  (avg_jan_to_jun * 6 - exp_jan + exp_jul) / 6 = 4250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_expenditure_feb_to_jul_l1347_134721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_18_or_21_but_not_both_l1347_134711

open Finset

theorem divisible_by_18_or_21_but_not_both : 
  (Finset.filter (λ n : Nat => n < 2019 ∧ 
    ((n % 18 = 0) ∨ (n % 21 = 0)) ∧ 
    ¬((n % 18 = 0) ∧ (n % 21 = 0))) 
    (Finset.range 2019)).card = 176 :=
by
  sorry

#check divisible_by_18_or_21_but_not_both

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_18_or_21_but_not_both_l1347_134711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_removal_always_possible_l1347_134737

/-- Represents a 4x4 grid with 6 stars --/
structure StarGrid where
  stars : Finset (Fin 4 × Fin 4)
  star_count : stars.card = 6
  distinct_cells : ∀ s₁ s₂, s₁ ∈ stars → s₂ ∈ stars → s₁ = s₂ ∨ s₁.1 ≠ s₂.1 ∨ s₁.2 ≠ s₂.2

/-- Two rows and two columns that can be removed --/
structure Removal where
  rows : Finset (Fin 4)
  cols : Finset (Fin 4)
  row_count : rows.card = 2
  col_count : cols.card = 2

/-- Theorem stating that for any valid StarGrid, there exists a Removal that eliminates all stars --/
theorem star_removal_always_possible (g : StarGrid) : 
  ∃ (r : Removal), ∀ s, s ∈ g.stars → s.1 ∈ r.rows ∨ s.2 ∈ r.cols := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_removal_always_possible_l1347_134737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_in_second_quadrant_implies_a_range_l1347_134714

def circle_eq (a x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*(a+1)*y + 3*a^2 + 3*a + 1 = 0

def second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem circle_in_second_quadrant_implies_a_range (a : ℝ) :
  (∀ x y, circle_eq a x y → second_quadrant x y) →
  0 < a ∧ a < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_in_second_quadrant_implies_a_range_l1347_134714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_in_base_6_l1347_134798

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 6 * acc) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def toBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The sum of 2015₆, 251₆, and 25₆ in base 6 is equal to 2335₆ -/
theorem sum_in_base_6 :
  toBase6 (toDecimal [2, 0, 1, 5] + toDecimal [2, 5, 1] + toDecimal [2, 5]) = [2, 3, 3, 5] := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_in_base_6_l1347_134798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_less_than_threshold_l1347_134751

noncomputable def numbers : List ℝ := [0.8, 1/2, 0.9, 1/3]
def threshold : ℝ := 0.4

theorem count_less_than_threshold : 
  (numbers.filter (λ x => x < threshold)).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_less_than_threshold_l1347_134751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_even_l1347_134776

/-- A function of the form f(x) = a*sin(x) + b*tan(x) + c -/
noncomputable def f (a b : ℝ) (c : ℤ) (x : ℝ) : ℝ := a * Real.sin x + b * Real.tan x + (c : ℝ)

/-- Theorem: The sum of f(2) and f(-2) is always an even integer -/
theorem f_sum_even (a b : ℝ) (c : ℤ) : 
  ∃ (k : ℤ), f a b c 2 + f a b c (-2) = 2 * k := by
  -- Proof sketch:
  -- 1. Define g(x) = a*sin(x) + b*tan(x)
  -- 2. Show that g is an odd function: g(-x) = -g(x)
  -- 3. Use this to show that g(2) + g(-2) = 0
  -- 4. Conclude that f(2) + f(-2) = 2c, which is an even integer
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_even_l1347_134776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_l1347_134719

theorem tan_period :
  ∃ T : ℝ, T > 0 ∧ 
  (∀ x : ℝ, Real.tan ((π / 2) * (x + T) - π / 3) = Real.tan ((π / 2) * x - π / 3)) ∧
  (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, Real.tan ((π / 2) * (x + S) - π / 3) = Real.tan ((π / 2) * x - π / 3)) → T ≤ S) ∧
  T = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_l1347_134719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1347_134744

noncomputable section

-- Define the fixed points A and B
def A : ℝ × ℝ := (-Real.sqrt 2, 0)
def B : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the slope product condition
def slope_product (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  (y / (x + Real.sqrt 2)) * (y / (x - Real.sqrt 2))

-- State the theorem
theorem trajectory_equation (P : ℝ × ℝ) :
  slope_product P = -1/2 → P.1^2 / 2 + P.2^2 = 1 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1347_134744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_two_sectors_90_deg_l1347_134782

/-- The area of a figure formed by two 90° sectors of a circle with radius 15 placed side by side -/
noncomputable def area_two_sectors (r : ℝ) (angle : ℝ) : ℝ :=
  2 * (angle / (2 * Real.pi)) * Real.pi * r^2

/-- Theorem: The area of a figure formed by two 90° sectors of a circle with radius 15 placed side by side is equal to 112.5π -/
theorem area_two_sectors_90_deg : 
  area_two_sectors 15 (Real.pi / 2) = 112.5 * Real.pi := by
  -- Expand the definition of area_two_sectors
  unfold area_two_sectors
  -- Simplify the expression
  simp [Real.pi]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_two_sectors_90_deg_l1347_134782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1347_134727

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / 4 + a / x - Real.log x - 3 / 2

theorem function_properties :
  ∃ (a : ℝ),
    (∀ x : ℝ, x > 0 → HasDerivAt (f a) ((1 / 4) - a / (x^2) - 1 / x) x) ∧
    (HasDerivAt (f a) (-2) 1) ∧
    (a = 5 / 4) ∧
    (∀ x : ℝ, 0 < x → x < 5 → (deriv (f a)) x < 0) ∧
    (∀ x : ℝ, x > 5 → (deriv (f a)) x > 0) ∧
    (IsLocalMin (f a) 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1347_134727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imaginary_part_of_roots_l1347_134718

theorem max_imaginary_part_of_roots (z : ℂ) (θ : ℝ) : 
  z^6 - z^4 + z^2 - 1 = 0 →
  -π/2 ≤ θ ∧ θ ≤ π/2 →
  (∃ (root : ℂ), root^6 - root^4 + root^2 - 1 = 0 ∧ 
    Complex.abs (root.im) ≤ Real.sin θ) →
  (∃ (max_root : ℂ), max_root^6 - max_root^4 + max_root^2 - 1 = 0 ∧ 
    Complex.abs (max_root.im) = Real.sin (π/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imaginary_part_of_roots_l1347_134718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_1_expression_evaluation_2_l1347_134773

-- Part 1
theorem expression_evaluation_1 :
  (0.064 : ℝ) ^ (-(1/3 : ℝ)) - (-1/8 : ℝ) ^ (0 : ℝ) + 16 ^ (3/4 : ℝ) + 0.25 ^ (1/2 : ℝ) = 10 := by sorry

-- Part 2
theorem expression_evaluation_2 :
  (1/2 : ℝ) * Real.log 25 + Real.log 2 - Real.log (Real.sqrt 0.1) - 
  (Real.log 9 / Real.log 2) * (Real.log 3 / Real.log 2) = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_1_expression_evaluation_2_l1347_134773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_transverse_axis_l1347_134766

/-- Represents a hyperbola with center at the origin -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  fociOnYAxis : Bool
  eccentricity : ℝ

/-- Represents a parabola -/
structure Parabola where
  p : ℝ

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem about the transverse axis of a hyperbola -/
theorem hyperbola_transverse_axis 
  (C : Hyperbola) 
  (P : Parabola)
  (h1 : C.fociOnYAxis = true)
  (h2 : C.eccentricity = Real.sqrt 2)
  (h3 : P.p = 2)
  (h4 : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ = -1 ∧ x₂ = -1 ∧ 
    C.a^2 * (y₁^2 / C.b^2 - x₁^2 / C.a^2) = 1 ∧
    C.a^2 * (y₂^2 / C.b^2 - x₂^2 / C.a^2) = 1 ∧
    distance x₁ y₁ x₂ y₂ = 4) :
  2 * C.a = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_transverse_axis_l1347_134766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_lies_on_line_l1347_134716

/-- A point (x, y) lies on a line if it satisfies the line equation y = mx + b,
    where m is the slope and b is the y-intercept. -/
def lies_on_line (x y m b : ℚ) : Prop :=
  y = m * x + b

/-- The slope of a line passing through two points (x₁, y₁) and (x₂, y₂). -/
def line_slope (x₁ y₁ x₂ y₂ : ℚ) : ℚ :=
  (y₂ - y₁) / (x₂ - x₁)

/-- The y-intercept of a line passing through (x, y) with slope m. -/
def y_intercept (x y m : ℚ) : ℚ :=
  y - m * x

theorem point_lies_on_line :
  let m := line_slope 1 4 3 7
  let b := y_intercept 1 4 m
  lies_on_line (-1/3) 2 m b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_lies_on_line_l1347_134716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_mx_iff_m_in_range_l1347_134767

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x^2 - x else x^2 - 3*x + 2

-- State the theorem
theorem f_geq_mx_iff_m_in_range :
  ∀ m : ℝ, (∀ x : ℝ, f x ≥ m * x) ↔ m ∈ Set.Icc (-3 - 2*Real.sqrt 2) 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_mx_iff_m_in_range_l1347_134767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_ratio_l1347_134708

/-- Represents a right circular cone -/
structure Cone where
  baseCircumference : ℝ
  height : ℝ

/-- The volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ :=
  (1/3) * c.baseCircumference^2 * c.height / (4 * Real.pi)

theorem cone_height_ratio (c : Cone) (h_new : ℝ) :
  c.baseCircumference = 20 * Real.pi →
  c.height = 25 →
  coneVolume { baseCircumference := c.baseCircumference, height := h_new } = 500 * Real.pi →
  h_new / c.height = 3 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_ratio_l1347_134708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1347_134723

theorem hyperbola_eccentricity_range (a b : ℝ) (lambda : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_lambda_lower : 5/12 ≤ lambda) (h_lambda_upper : lambda ≤ 4/3) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  ∃ (P Q F₁ F₂ : ℝ × ℝ),
    (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
    (Q.1^2 / a^2 - Q.2^2 / b^2 = 1) ∧
    (F₂.1 > F₁.1) ∧
    (∃ (k : ℝ), Q = k • (P - F₂) + F₂) ∧
    ((P - Q).1 * (P - F₁).1 + (P - Q).2 * (P - F₁).2 = 0) ∧
    (Real.sqrt ((P - Q).1^2 + (P - Q).2^2) = lambda * Real.sqrt ((P - F₁).1^2 + (P - F₁).2^2)) →
    Real.sqrt (37/25) ≤ e ∧ e ≤ Real.sqrt (5/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1347_134723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_fixed_point_l1347_134771

-- Define the Cartesian plane
variable (x y : ℝ)

-- Define points P and M
def P : ℝ × ℝ := (x, y)
def M : ℝ × ℝ := (x, -4)

-- Define the condition that O is on the circle with diameter PM
def circle_condition (x y : ℝ) : Prop :=
  (x^2 + y^2) * (x^2 + (-4-y)^2) = 4 * ((x^2 + y*(-4))^2 + (y-(-4))^2 * x^2)

-- Define the trajectory W
def W : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 = 4*p.2}

-- Define a line passing through E(0,-4)
def line_through_E (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k*p.1 - 4}

-- Define the reflection of a point across the y-axis
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Theorem 1: The trajectory W satisfies x^2 = 4y
theorem trajectory_equation (x y : ℝ) : 
  circle_condition x y → (x, y) ∈ W := by sorry

-- Theorem 2: A'B always passes through (0,4)
theorem fixed_point (k : ℝ) (A B : ℝ × ℝ) :
  A ∈ W → B ∈ W → A ∈ line_through_E k → B ∈ line_through_E k →
  ∃ (m c : ℝ), (0, 4) ∈ {p : ℝ × ℝ | p.2 = m*p.1 + c} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_fixed_point_l1347_134771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_value_l1347_134722

theorem cos_2beta_value (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.tan α = 1/7 →
  Real.cos (α + β) = 2 * Real.sqrt 5 / 5 →
  Real.cos (2 * β) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_value_l1347_134722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l1347_134785

noncomputable section

variable (a : ℝ)

def z : ℂ := (a + Complex.I) / (2 * Complex.I)

theorem modulus_of_z : ∀ (a : ℝ), (z a).re = (z a).im → Complex.abs (z a) = Real.sqrt 2 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l1347_134785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_ratio_l1347_134728

theorem product_sum_ratio : (Finset.prod (Finset.range 10) (λ i => i + 1)) / (Finset.sum (Finset.range 10) (λ i => i + 1)) = 660 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_ratio_l1347_134728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1347_134757

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) + 2 * Real.sin x ^ 2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M) ∧ M = 2) ∧
  (∃ (m : ℝ), (∀ (x : ℝ), m ≤ f x) ∧ (∃ (x : ℝ), f x = m) ∧ m = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1347_134757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strict_boundary_and_max_integer_l1347_134740

-- Define the domain
def D : Set ℝ := {x | -1 < x ∧ x < 0}

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 1 + x
noncomputable def g (x : ℝ) : ℝ := 1 + x + x^2 / 2
noncomputable def F (x : ℝ) : ℝ := Real.exp x
noncomputable def h (x : ℝ) : ℝ := 2 * Real.exp x + 1 / (1 + x) - 2

-- State the theorem
theorem strict_boundary_and_max_integer :
  (∀ x ∈ D, f x < F x ∧ F x < g x) ∧
  (∃ M : ℤ, (∀ x ∈ D, h x > M / 10) ∧
            (∀ N : ℤ, N > M → ∃ x ∈ D, h x ≤ N / 10) ∧
            M = 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strict_boundary_and_max_integer_l1347_134740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l1347_134791

/-- Line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

/-- Curve C in the xy-plane -/
def curve_C (x y : ℝ) : Prop := x^2 + y^2/3 = 1

/-- Distance from a point (x, y) to line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x - y + 3| / Real.sqrt 2

/-- Theorem stating the range of distances from curve C to line l -/
theorem distance_range :
  ∀ x y : ℝ, curve_C x y →
  ∃ d : ℝ, d = distance_to_line x y ∧
  Real.sqrt 2 / 2 ≤ d ∧ d ≤ 5 * Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l1347_134791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_l1347_134780

noncomputable section

open Real

-- Define the two functions
def f (x : ℝ) : ℝ := sin (π / 2 + 2 * x)
def g (x : ℝ) : ℝ := cos (2 * x - π / 3)

-- Theorem stating the relationship between the two functions
theorem graph_shift (x : ℝ) : f x = g (x - π / 6) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_l1347_134780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_constant_l1347_134700

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h : a > b
  k : b > 0

/-- A point on the ellipse -/
def PointOnEllipse (e : Ellipse a b) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The left focus of the ellipse -/
noncomputable def leftFocus (e : Ellipse a b) : ℝ × ℝ := (-Real.sqrt (a^2 - b^2), 0)

/-- The right focus of the ellipse -/
noncomputable def rightFocus (e : Ellipse a b) : ℝ × ℝ := (Real.sqrt (a^2 - b^2), 0)

/-- The left vertex of the ellipse -/
def leftVertex (a b : ℝ) : ℝ × ℝ := (-a, 0)

/-- The right vertex of the ellipse -/
def rightVertex (a b : ℝ) : ℝ × ℝ := (a, 0)

/-- The x-coordinate of the right directrix -/
noncomputable def rightDirectrixX (e : Ellipse a b) : ℝ := a^2 / Real.sqrt (a^2 - b^2)

/-- The intersection of PA with the right directrix -/
noncomputable def intersectionPA (e : Ellipse a b) (x₀ y₀ : ℝ) : ℝ × ℝ :=
  let x := rightDirectrixX e
  let y := y₀ * (x + a) / (x₀ + a)
  (x, y)

/-- The intersection of PB with the right directrix -/
noncomputable def intersectionPB (e : Ellipse a b) (x₀ y₀ : ℝ) : ℝ × ℝ :=
  let x := rightDirectrixX e
  let y := y₀ * (x - a) / (x₀ - a)
  (x, y)

/-- The dot product of two 2D vectors -/
def dotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The main theorem: for any point on the ellipse, MF₁ · NF₂ = 2b² -/
theorem ellipse_dot_product_constant (e : Ellipse a b) (x₀ y₀ : ℝ) 
  (h : PointOnEllipse e x₀ y₀) :
  let M := intersectionPA e x₀ y₀
  let N := intersectionPB e x₀ y₀
  let F₁ := leftFocus e
  let F₂ := rightFocus e
  dotProduct (M.1 - F₁.1, M.2 - F₁.2) (N.1 - F₂.1, N.2 - F₂.2) = 2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_constant_l1347_134700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_percent_per_annum_l1347_134707

-- Define the given values
noncomputable def principal : ℝ := 2000
noncomputable def interest_increase : ℝ := 40
noncomputable def time_increase : ℝ := 4

-- Define the simple interest formula
noncomputable def simple_interest (p r t : ℝ) : ℝ := p * r * t / 100

-- State the theorem
theorem rate_percent_per_annum :
  ∃ (r : ℝ),
    simple_interest principal r time_increase = interest_increase ∧
    r = 0.5 := by
  -- Provide the value of r
  use 0.5
  -- Split the goal into two parts
  constructor
  -- Prove the first part: simple_interest principal 0.5 time_increase = interest_increase
  · simp [simple_interest, principal, time_increase, interest_increase]
    norm_num
  -- Prove the second part: 0.5 = 0.5
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_percent_per_annum_l1347_134707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_harmonic_progression_l1347_134752

noncomputable def floor (x : ℝ) := ⌊x⌋

noncomputable def frac (x : ℝ) : ℝ := x - floor x

def isHarmonicProgression (a b c : ℝ) : Prop :=
  2 / b = 1 / a + 1 / c

theorem two_solutions_harmonic_progression :
  ∃! (s : Finset ℝ), s.card = 2 ∧
    ∀ x ∈ s, isHarmonicProgression x (floor x) (frac x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_harmonic_progression_l1347_134752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_equations_l1347_134736

noncomputable def x (t : ℝ) : ℝ := t * (t * Real.cos t - 2 * Real.sin t)
noncomputable def y (t : ℝ) : ℝ := t * (t * Real.sin t + 2 * Real.cos t)

noncomputable def t₀ : ℝ := Real.pi / 4

theorem tangent_and_normal_equations :
  let x₀ := x t₀
  let y₀ := y t₀
  let slope := -(Real.cos t₀ / Real.sin t₀)
  (∀ x y : ℝ, y = slope * (x - x₀) + y₀ ↔ y = -x + Real.pi^2 * Real.sqrt 2 / 16) ∧
  (∀ x y : ℝ, y = -(1/slope) * (x - x₀) + y₀ ↔ y = x + Real.pi * Real.sqrt 2 / 2) := by
  sorry

#check tangent_and_normal_equations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_equations_l1347_134736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_with_sqrt5_minus3_root_l1347_134710

theorem quadratic_equation_with_sqrt5_minus3_root :
  ∃ (a b c : ℚ), a ≠ 0 ∧
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3) ∧
  a * (Real.sqrt 5 - 3)^2 + b * (Real.sqrt 5 - 3) + c = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_with_sqrt5_minus3_root_l1347_134710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_integer_count_l1347_134783

def sequence_term (n : ℕ) : ℚ :=
  9720 / 3^n

def is_integer (q : ℚ) : Prop :=
  ∃ k : ℤ, q = k

theorem sequence_integer_count :
  (∃ n : ℕ, n > 0 ∧ ∀ k < n, is_integer (sequence_term k) ∧ ¬is_integer (sequence_term n)) →
  (∃! n : ℕ, n = 6 ∧ ∀ k < n, is_integer (sequence_term k) ∧ ¬is_integer (sequence_term n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_integer_count_l1347_134783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_cut_sum_l1347_134799

-- Define the cube edge length
def cube_edge : ℝ := 3

-- Define the pyramidal segment volume
noncomputable def pyramidal_volume : ℝ := (1/3) * (9/4) * cube_edge

-- Define the iced surface area
noncomputable def iced_surface_area : ℝ := (9/4) + 3 * ((9 * Real.sqrt 13) / 8)

-- Theorem statement
theorem cube_cut_sum :
  pyramidal_volume + iced_surface_area = 9/2 + (27 * Real.sqrt 13) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_cut_sum_l1347_134799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l1347_134706

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- Define the domain
def D : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Theorem statement
theorem f_odd_and_increasing : 
  (∀ x, x ∈ D → f (-x) = -f x) ∧ 
  (∀ x y, x ∈ D → y ∈ D → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l1347_134706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1347_134781

-- Define the ellipse C₁
def C₁ (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the hyperbola C₂
def C₂ (x y : ℝ) : Prop := x^2 - y^2 = 4

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- State the theorem
theorem ellipse_eccentricity 
  (a b : ℝ) 
  (h₁ : a > b) 
  (h₂ : b > 0) 
  (x y : ℝ) 
  (h₃ : C₁ x y a b) 
  (h₄ : C₂ x y) 
  (xf yf : ℝ) -- Coordinates of the right focus F₂
  (h₅ : distance x y xf yf = 2) :
  a^2 - b^2 = 8 := by sorry

-- The eccentricity is defined as e = √(a² - b²) / a
-- So, if a² - b² = 8, then e = √8 / 4 = √2 / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1347_134781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1347_134786

noncomputable def f (x : ℝ) : ℝ := x + 1/x + (x^2 + 1/x^2) + 1/(x + 1/x + (x^2 + 1/x^2))

theorem f_minimum_value :
  (∀ x > 0, f x ≥ 4.25) ∧ (∃ x > 0, f x = 4.25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1347_134786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_raise_approx_l1347_134775

/-- Represents a salary raise range with lower and upper bounds -/
structure RaiseRange where
  lower : Float
  upper : Float
  lower_lt_upper : lower < upper

/-- Calculates the midpoint of a raise range -/
def RaiseRange.midpoint (r : RaiseRange) : Float :=
  (r.lower + r.upper) / 2

/-- The raise range for Renee -/
def renee_raise : RaiseRange where
  lower := 5
  upper := 10
  lower_lt_upper := by sorry

/-- The raise range for Sophia -/
def sophia_raise : RaiseRange where
  lower := 7
  upper := 12
  lower_lt_upper := by sorry

/-- The raise range for Carlos -/
def carlos_raise : RaiseRange where
  lower := 4
  upper := 9
  lower_lt_upper := by sorry

/-- Calculates the average of the midpoints of the three raise ranges -/
def average_raise : Float :=
  (renee_raise.midpoint + sophia_raise.midpoint + carlos_raise.midpoint) / 3

/-- Theorem stating that the average raise is approximately 7.83% -/
theorem average_raise_approx :
  (average_raise - 7.83).abs < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_raise_approx_l1347_134775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_percentage_l1347_134703

noncomputable def weight_scheme (w : ℝ) : ℝ :=
  if w < 5 then 0.852
  else if w < 10 then 0.910
  else if w < 20 then 0.960
  else 0.993

def customer_purchases : List ℝ := [3, 6, 12, 22]

noncomputable def actual_weight (w : ℝ) : ℝ := w * weight_scheme w

noncomputable def total_should_receive : ℝ := customer_purchases.sum

noncomputable def total_actually_receive : ℝ := (customer_purchases.map actual_weight).sum

noncomputable def profit : ℝ := total_should_receive - total_actually_receive

noncomputable def profit_percentage : ℝ := (profit / total_should_receive) * 100

theorem dealer_profit_percentage :
  abs (profit_percentage - 3.76) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_percentage_l1347_134703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_passes_through_fixed_point_l1347_134734

/-- A line in the plane represented by its slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The symmetric line of l1 with respect to y = x - 1 -/
noncomputable def symmetricLine (l1 : Line) : Line :=
  { slope := 1 / l1.slope, intercept := 3 - 1 / l1.slope }

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- The fixed point (3, 0) -/
def fixedPoint : Point := { x := 3, y := 0 }

/-- The main theorem -/
theorem symmetric_line_passes_through_fixed_point (k : ℝ) :
  let l1 : Line := { slope := k, intercept := 2 - k }
  let l2 := symmetricLine l1
  pointOnLine fixedPoint l2 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_passes_through_fixed_point_l1347_134734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_B_l1347_134750

theorem polynomial_coefficient_B : ∀ (A C D : ℤ),
  let p : Polynomial ℤ := X^6 - 10*X^5 + A*X^4 - 88*X^3 + C*X^2 + D*X + 16
  ∀ (roots : Finset ℤ),
    (∀ r ∈ roots, r > 0) →
    (roots.sum id = 10) →
    (roots.card = 6) →
    p.roots = roots.val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_B_l1347_134750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_maximizes_length_l1347_134777

/-- Two circles in a plane -/
structure TwoCircles where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ
  radius1 : ℝ
  radius2 : ℝ
  intersection : ℝ × ℝ

/-- A secant line passing through the intersection point of two circles -/
structure Secant where
  direction : ℝ × ℝ  -- Direction vector of the secant line
  pointA : ℝ × ℝ     -- Intersection with first circle
  pointB : ℝ × ℝ     -- Intersection with second circle

/-- The length of a segment given its endpoints -/
noncomputable def segmentLength (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

/-- The direction vector from one point to another -/
def directionVector (a b : ℝ × ℝ) : ℝ × ℝ :=
  (b.1 - a.1, b.2 - a.2)

/-- Check if two vectors are perpendicular -/
def isPerpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

/-- Theorem: The secant perpendicular to the line joining the centers maximizes the length of AB -/
theorem secant_maximizes_length (c : TwoCircles) :
  ∀ (s : Secant),
    isPerpendicular (directionVector c.center1 c.center2) s.direction →
    ∀ (s' : Secant),
      segmentLength s.pointA s.pointB ≥ segmentLength s'.pointA s'.pointB := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_maximizes_length_l1347_134777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_pi_div_2_is_odd_and_pi_periodic_l1347_134774

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 2)

theorem cos_2x_plus_pi_div_2_is_odd_and_pi_periodic :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ T, (0 < T ∧ T < Real.pi) → ∃ x, f (x + T) ≠ f x) := by
  sorry

#check cos_2x_plus_pi_div_2_is_odd_and_pi_periodic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_pi_div_2_is_odd_and_pi_periodic_l1347_134774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1347_134754

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin (x / 2) * Real.cos (x / 2) + 2 * (Real.cos (x / 2))^2

theorem f_properties :
  -- 1. Smallest positive period is 2π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (let p := Real.pi * 2; p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- 2. Monotonically decreasing in the specified interval
  (∀ (k : ℤ) (x y : ℝ), 2 * ↑k * Real.pi + Real.pi / 3 ≤ x ∧ x < y ∧ y ≤ 2 * ↑k * Real.pi + 4 * Real.pi / 3 → f y < f x) ∧
  -- 3. Triangle ABC properties
  (∀ (A B C : ℝ) (a b c : ℝ),
    f B = 3 →
    b = 3 →
    Real.sin C = 2 * Real.sin A →
    a = Real.sqrt 3 →
    c = 2 * Real.sqrt 3 →
    a * Real.sin C = c * Real.sin A ∧
    b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1347_134754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1347_134753

/-- Given vectors a and b in ℝ², prove that if |a| = 1, b = (1/2, m), and (a + b) ⟂ (a - b), then m = ± √3/2 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (m : ℝ) :
  (a.1^2 + a.2^2 = 1) →  -- |a| = 1
  (b = (1/2, m)) →       -- b = (1/2, m)
  ((a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2) = 0) →  -- (a + b) ⟂ (a - b)
  (m = Real.sqrt 3 / 2 ∨ m = -Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1347_134753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_find_a_b_l1347_134746

-- Define sets A and B
def A : Set ℝ := {x | (1/2 : ℝ)^(x^2 - 4) > 1}
def B : Set ℝ := {x | 2 < 4/(x + 3)}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | -2 < x ∧ x < 1} := by sorry

-- Define the quadratic inequality
def quadratic_inequality (a b : ℝ) : Set ℝ := {x | 2*x^2 + a*x + b < 0}

-- Theorem for part (2)
theorem find_a_b : ∃ (a b : ℝ), quadratic_inequality a b = B ∧ a = 4 ∧ b = -6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_find_a_b_l1347_134746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1347_134712

noncomputable def f (x : ℝ) : ℝ := 3 * (Real.sin x)^3 - 7 * (Real.sin x)^2 + 4 * Real.sin x

theorem equation_solutions :
  ∃! (s : Finset ℝ), s.card = 4 ∧ 
  (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = 0) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = 0 → x ∈ s) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1347_134712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abc_equals_four_l1347_134701

theorem sum_abc_equals_four (A B C : ℕ+) : 
  (Nat.gcd A.val (Nat.gcd B.val C.val) = 1) →
  (A : ℝ) * (Real.log 3 / Real.log 180) + (B : ℝ) * (Real.log 5 / Real.log 180) = C →
  A + B + C = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abc_equals_four_l1347_134701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_of_rectangle_system_l1347_134726

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the length of the diagonal of a rectangle using the Pythagorean theorem -/
noncomputable def Rectangle.diagonal (r : Rectangle) : ℝ := Real.sqrt (r.width ^ 2 + r.height ^ 2)

/-- Represents a system of four rectangles with specific properties -/
structure RectangleSystem where
  R₀ : Rectangle
  R₁ : Rectangle
  R₂ : Rectangle
  R₃ : Rectangle
  common_vertex : Bool
  diagonal_side : Bool
  counterclockwise : Bool

/-- Theorem stating the total area covered by the union of four rectangles in the system -/
theorem total_area_of_rectangle_system (rs : RectangleSystem) : 
  rs.R₀.width = 3 ∧ rs.R₀.height = 4 ∧ 
  rs.common_vertex ∧ rs.diagonal_side ∧ rs.counterclockwise →
  rs.R₀.area + rs.R₁.area + rs.R₂.area + rs.R₃.area - 
  (1/2 * rs.R₁.area + 1/2 * rs.R₂.area + 1/2 * rs.R₃.area) = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_of_rectangle_system_l1347_134726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_mixture_cost_l1347_134739

/-- Represents the cost per pound of the first candy -/
def C : ℝ := sorry

/-- The weight of the first candy in pounds -/
def weight_first : ℝ := 20

/-- The weight of the second candy in pounds -/
def weight_second : ℝ := 40

/-- The cost per pound of the second candy -/
def cost_second : ℝ := 5

/-- The cost per pound of the mixture -/
def cost_mixture : ℝ := 6

/-- The total weight of the mixture -/
def total_weight : ℝ := weight_first + weight_second

theorem candy_mixture_cost :
  weight_first * C + weight_second * cost_second = total_weight * cost_mixture →
  C = 8 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_mixture_cost_l1347_134739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_15_terms_eq_202_5_l1347_134735

/-- An arithmetic progression with specified third and fifth terms -/
structure ArithmeticProgression where
  a3 : ℚ
  a5 : ℚ

/-- The sum of the first n terms of an arithmetic progression -/
noncomputable def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  let d := (ap.a5 - ap.a3) / 2
  let a1 := ap.a3 - 2 * d
  (n : ℚ) / 2 * (2 * a1 + (n - 1 : ℚ) * d)

/-- Theorem: The sum of the first 15 terms of the specified arithmetic progression is 202.5 -/
theorem sum_15_terms_eq_202_5 (ap : ArithmeticProgression) 
    (h1 : ap.a3 = -5) (h2 : ap.a5 = 12/5) : 
    sum_n_terms ap 15 = 405/2 := by
  sorry

#eval (405 : ℚ) / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_15_terms_eq_202_5_l1347_134735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_half_l1347_134729

noncomputable def f (ω b : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4) + b

theorem f_value_at_pi_half 
  (ω b : ℝ) 
  (h_ω_pos : ω > 0)
  (h_period : 2 * Real.pi / 3 < 2 * Real.pi / ω ∧ 2 * Real.pi / ω < Real.pi)
  (h_symmetry : ∀ x, f ω b (3 * Real.pi / 2 - x) = f ω b (3 * Real.pi / 2 + x))
  : f ω b (Real.pi / 2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_half_l1347_134729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_range_f_odd_l1347_134702

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - x^4) / (abs (x - 1) - 1)

-- State the theorems
theorem f_domain : 
  {x : ℝ | f x ≠ 0} = Set.Icc (-1 : ℝ) 0 ∪ Set.Ioc 0 1 := by sorry

theorem f_range : 
  Set.range f = Set.Ioo (-1 : ℝ) 1 := by sorry

theorem f_odd : 
  ∀ x, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_range_f_odd_l1347_134702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_10_plus_1_int_part_sqrt_11_expression_l1347_134761

-- Define the integer part function
noncomputable def intPart (x : ℝ) : ℤ := ⌊x⌋

-- Statement 1
theorem sqrt_10_plus_1_int_part : intPart (Real.sqrt 10 + 1) = 4 := by sorry

-- Statement 2
theorem sqrt_11_expression :
  let a : ℤ := intPart (Real.sqrt 11)
  let b : ℝ := Real.sqrt 11 - (a : ℝ)
  let c : ℝ := Real.sqrt 11  -- We choose the positive value for c
  c * ((a : ℝ) - b - 6) + 12 = 1 ∨ c * ((a : ℝ) - b - 6) + 12 = 23 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_10_plus_1_int_part_sqrt_11_expression_l1347_134761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f₁_is_odd_l1347_134789

-- Define the four functions
noncomputable def f₁ (x : ℝ) : ℝ := 1 / x
def f₂ (x : ℝ) : ℝ := |x|
noncomputable def f₃ (x : ℝ) : ℝ := Real.log x / Real.log 10
def f₄ (x : ℝ) : ℝ := x^3 + 1

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem only_f₁_is_odd :
  is_odd f₁ ∧ ¬is_odd f₂ ∧ ¬is_odd f₃ ∧ ¬is_odd f₄ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f₁_is_odd_l1347_134789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_k_value_l1347_134743

/-- Given two vectors a and b in ℝ³, prove that if ka + b and a + kb are parallel,
    then k = 1 or k = -1. -/
theorem parallel_vectors_k_value (k : ℝ) :
  let a : Fin 3 → ℝ := ![1, 1, 0]
  let b : Fin 3 → ℝ := ![-1, 0, 2]
  (∃ (c : ℝ), k • a + b = c • (a + k • b)) →
  k = 1 ∨ k = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_k_value_l1347_134743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_volume_l1347_134784

/-- The volume of the mixture obtained by combining 3/5 of 20 liters of water and 5/6 of 18 liters of vinegar is 27 liters. -/
theorem mixture_volume (water vinegar : ℝ)
  (h_water : water = 20 * (3 / 5))
  (h_vinegar : vinegar = 18 * (5 / 6))
  : water + vinegar = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_volume_l1347_134784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l1347_134794

-- Define the circle
def circle_P (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the point M
def point_M : ℝ × ℝ := (-1, 2)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - 2*y + 5 = 0

-- Theorem statement
theorem tangent_line_proof :
  ∃! (k : ℝ), 
    (∀ x y : ℝ, tangent_line x y ↔ y - point_M.2 = k * (x - point_M.1)) ∧
    (∀ x y : ℝ, circle_P x y → (x - point_M.1)^2 + (y - point_M.2)^2 ≥ (x^2 + y^2 - point_M.1^2 - point_M.2^2) / 4) ∧
    (∃ x y : ℝ, circle_P x y ∧ tangent_line x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l1347_134794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1347_134738

def S (n : ℕ) : ℤ := 33 * n - n^2

def a (n : ℕ) : ℤ := 34 - 2 * n

def b (n : ℕ) : ℕ := Int.natAbs (a n)

def S' (n : ℕ) : ℕ := 
  if n ≤ 17 then
    Int.natAbs (33 * n - n^2)
  else
    (n^2 - 33 * n + 544)

theorem sequence_properties :
  (∀ n : ℕ, S (n + 1) - S n = a (n + 1)) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 34 - 2 * n) ∧
  (∀ n : ℕ, n ≥ 18 → S n ≤ S 17) ∧
  (∀ n : ℕ, n ≤ 17 → S' n = Int.natAbs (S n)) ∧
  (∀ n : ℕ, n ≥ 18 → S' n = n^2 - 33 * n + 544) := by
  sorry

#eval S' 17
#eval S' 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1347_134738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_two_l1347_134717

-- Define the derivative of f as a separate function
noncomputable def f' (x : ℝ) : ℝ := 2*x - 4

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x*(f' 1)

-- State the theorem
theorem derivative_at_two : 
  (∀ x, HasDerivAt f (f' x) x) → f' 2 = 0 := by
  intro h
  -- The proof goes here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_two_l1347_134717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_start_cities_l1347_134788

-- Define the graph structure
structure CityGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  vertex_count : vertices.card = 6
  edge_count : edges.card = 6

-- Define a valid path in the graph
def ValidPath (g : CityGraph) (path : List Nat) : Prop :=
  path.length = g.vertices.card ∧
  path.toFinset = g.vertices ∧
  ∀ i, i + 1 < path.length → (path[i]!, path[i+1]!) ∈ g.edges ∨ (path[i+1]!, path[i]!) ∈ g.edges

-- Define the degree of a vertex
def Degree (g : CityGraph) (v : Nat) : Nat :=
  (g.edges.filter (λ e => e.1 = v ∨ e.2 = v)).card

-- Theorem statement
theorem valid_start_cities (g : CityGraph) :
  ∀ path : List Nat, ValidPath g path →
    (path ≠ [] → Degree g (path.head!) = 1 ∧ Degree g (path.getLast!) = 1) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_start_cities_l1347_134788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_internally_l1347_134749

/-- Circle represented by its general equation -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- Center of a circle -/
noncomputable def center (c : Circle) : ℝ × ℝ :=
  (- c.a / 2, - c.b / 2)

/-- Radius of a circle -/
noncomputable def radius (c : Circle) : ℝ :=
  Real.sqrt ((c.a / 2)^2 + (c.b / 2)^2 - c.e)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Two circles are tangent internally if the distance between their centers
    equals the difference of their radii -/
def tangent_internally (c1 c2 : Circle) : Prop :=
  distance (center c1) (center c2) = abs (radius c2 - radius c1)

theorem circles_tangent_internally :
  let c1 : Circle := { a := -4, b := -6, c := 1, d := 1, e := 12 }
  let c2 : Circle := { a := -8, b := -6, c := 1, d := 1, e := 16 }
  tangent_internally c1 c2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_internally_l1347_134749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1347_134796

-- Define the circle in polar coordinates
def circle_polar (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi / 4) + 6 = 0

-- Define the Cartesian equation
def cartesian_eq (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 2

-- Define the parametric equations
def parametric_eq (x y α : ℝ) : Prop :=
  x = 2 + Real.sqrt 2 * Real.cos α ∧ y = 2 + Real.sqrt 2 * Real.sin α

-- Theorem stating the equivalence and extrema
theorem circle_properties :
  (∀ ρ θ, circle_polar ρ θ ↔ ∃ x y, cartesian_eq x y ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  (∀ x y, cartesian_eq x y ↔ ∃ α, parametric_eq x y α) ∧
  (∀ x y, cartesian_eq x y → x + y ≤ 6) ∧
  (∀ x y, cartesian_eq x y → x + y ≥ 2) ∧
  (∃ x y, cartesian_eq x y ∧ x + y = 6) ∧
  (∃ x y, cartesian_eq x y ∧ x + y = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1347_134796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jamie_coin_jar_l1347_134778

/-- Represents the number of each type of coin in the jar -/
def num_coins : ℕ := 34

/-- The total value of coins in cents -/
def total_value : ℕ := 3100

/-- The equation representing the total value of coins -/
axiom coin_value_eq : 1 * num_coins + 5 * num_coins + 10 * num_coins + 25 * num_coins + 50 * num_coins = total_value

/-- The number of different types of coins -/
def num_types : ℕ := 5

theorem jamie_coin_jar :
  num_coins = 34 ∧ num_types * num_coins = 170 := by
  constructor
  · rfl
  · rfl

#eval num_types * num_coins -- To verify the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jamie_coin_jar_l1347_134778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_sum_of_squares_is_1003_l1347_134787

/-- The expression to be simplified -/
def original_expression (x : ℝ) : ℝ := 3 * (x^2 - 3*x + 3) - 8 * (x^3 - 2*x^2 + x - 1)

/-- The fully simplified form of the expression -/
def simplified_expression (x : ℝ) : ℝ := -8*x^3 + 19*x^2 - 17*x + 17

/-- Theorem stating that the original expression equals the simplified expression -/
theorem expression_equality (x : ℝ) : original_expression x = simplified_expression x := by sorry

/-- The sum of squares of the coefficients of the simplified expression -/
def sum_of_squares : ℕ := 64 + 361 + 289 + 289

/-- Theorem proving that the sum of squares of the coefficients is 1003 -/
theorem sum_of_squares_is_1003 : sum_of_squares = 1003 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_sum_of_squares_is_1003_l1347_134787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1347_134713

noncomputable def a : ℝ := Real.pi ^ (-2 : ℝ)
noncomputable def b : ℝ := a ^ a
noncomputable def c : ℝ := a ^ (a ^ a)

theorem relationship_abc : b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1347_134713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_closer_to_hillcreast_l1347_134745

noncomputable section

-- Define the distance between towns
def total_distance : ℝ := 120

-- Define the speed of the person from Hillcreast
def hillcreast_speed : ℝ := 5

-- Define the initial speed of the person from Sunview
def sunview_initial_speed : ℝ := 4

-- Define the speed increase rate for the person from Sunview
def sunview_speed_increase : ℝ := 0.5  -- 1 mile per hour every 2 hours

-- Define the function for the distance traveled by the person from Hillcreast
def hillcreast_distance (t : ℝ) : ℝ := hillcreast_speed * t

-- Define the function for the distance traveled by the person from Sunview
def sunview_distance (t : ℝ) : ℝ := 
  sunview_initial_speed * t + sunview_speed_increase * t^2 / 2

-- Theorem to prove
theorem meeting_point_closer_to_hillcreast :
  ∃ (t : ℕ), 
    hillcreast_distance (t : ℝ) + sunview_distance (t : ℝ) = total_distance ∧
    10 ≤ total_distance - 2 * hillcreast_distance (t : ℝ) ∧
    total_distance - 2 * hillcreast_distance (t : ℝ) < 11 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_closer_to_hillcreast_l1347_134745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1347_134795

theorem power_equality (y : ℚ) (h : (128 : ℝ)^(7 : ℝ) = (32 : ℝ)^(y : ℝ)) : 
  (2 : ℝ)^((-3 : ℝ) * (y : ℝ)) = 1 / ((2 : ℝ)^((147 : ℝ)/(5 : ℝ))) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1347_134795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cos_a_l1347_134790

theorem largest_cos_a (a b c : ℝ) 
  (h1 : Real.sin a = 1 / Real.tan b)
  (h2 : Real.sin b = 1 / Real.tan c)
  (h3 : Real.sin c = 1 / Real.tan a) :
  ∃ (max_cos_a : ℝ), 
    (∀ x, x = Real.cos a → x ≤ max_cos_a) ∧ 
    max_cos_a = Real.sqrt ((3 - Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cos_a_l1347_134790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l1347_134756

theorem sum_remainder_mod_15 (d e f : ℕ) 
  (hd : d % 15 = 11)
  (he : e % 15 = 12)
  (hf : f % 15 = 13) :
  (d + e + f) % 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l1347_134756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_H_is_unit_circle_l1347_134748

-- Define the line l
noncomputable def line_l (φ : ℝ) : ℝ × ℝ → Prop :=
  λ p => ∃ t, p.1 = t * Real.cos φ ∧ p.2 = -1 + t * Real.sin φ

-- Define point B
def point_B : ℝ × ℝ := (0, 1)

-- Define the foot of the perpendicular H
noncomputable def foot_H (φ : ℝ) : ℝ × ℝ :=
  ((2 * Real.tan φ) / (1 + Real.tan φ * Real.tan φ),
   (Real.tan φ * Real.tan φ - 1) / (1 + Real.tan φ * Real.tan φ))

-- Theorem: The trajectory of H is a unit circle
theorem trajectory_H_is_unit_circle :
  ∀ φ, (foot_H φ).1 ^ 2 + (foot_H φ).2 ^ 2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_H_is_unit_circle_l1347_134748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_f_iter_eq_f_l1347_134731

def f (x : ℕ) : ℕ :=
  if x % 3 = 0 ∧ x % 7 = 0 then x / 21
  else if x % 7 = 0 then 3 * x
  else if x % 3 = 0 then 7 * x
  else x + 3

def f_iter : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => f (f_iter n x)

theorem smallest_a_for_f_iter_eq_f :
  ∃ a : ℕ, a > 1 ∧ f_iter a 2 = f 2 ∧ ∀ k, 1 < k → k < a → f_iter k 2 ≠ f 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_f_iter_eq_f_l1347_134731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_45_45_90_triangle_hypotenuse_length_specific_triangle_l1347_134763

/-- In a 45-45-90 triangle with an inscribed circle of radius r,
    the length of the hypotenuse is 4r√2. --/
theorem hypotenuse_length_45_45_90_triangle (r : ℝ) (h : r > 0) :
  let triangle := {(a, b, c) : ℝ × ℝ × ℝ | a = b ∧ a^2 + b^2 = c^2}
  let inscribed_circle_radius := r
  let hypotenuse_length := 4 * r * Real.sqrt 2
  (r, r, hypotenuse_length) ∈ triangle ∧
  inscribed_circle_radius = r →
  hypotenuse_length = 4 * r * Real.sqrt 2 :=
by sorry

/-- The length of the hypotenuse of a 45-45-90 triangle
    with an inscribed circle of radius 8 cm is 16√2 cm. --/
theorem hypotenuse_length_specific_triangle :
  let r : ℝ := 8
  let hypotenuse_length := 4 * r * Real.sqrt 2
  hypotenuse_length = 16 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_45_45_90_triangle_hypotenuse_length_specific_triangle_l1347_134763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_siding_cost_l1347_134759

/-- Calculates the total cost of siding required for a structure with given dimensions -/
noncomputable def total_siding_cost (wall_width wall_height roof_base roof_height siding_width siding_height siding_cost : ℝ) : ℝ :=
  let wall_area := 2 * (wall_width * wall_height)
  let roof_area := 2 * (0.5 * roof_base * roof_height)
  let total_area := wall_area + roof_area
  let siding_area := siding_width * siding_height
  let sections_needed := ⌈total_area / siding_area⌉
  sections_needed * siding_cost

/-- The total cost of siding for Sandy's structure is $70 -/
theorem sandy_siding_cost :
  total_siding_cost 10 6 10 7 10 15 35 = 70 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_siding_cost_l1347_134759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_imaginary_roots_necessary_not_sufficient_l1347_134733

theorem quadratic_imaginary_roots (a : ℝ) : 
  (∀ x : ℂ, x^2 + a*x + 1 = 0 → x.im ≠ 0) ↔ -2 < a ∧ a < 2 :=
sorry

theorem necessary_not_sufficient : 
  (∀ a : ℝ, (∀ x : ℂ, x^2 + a*x + 1 = 0 → x.im ≠ 0) → -2 ≤ a ∧ a ≤ 2) ∧
  ¬(∀ a : ℝ, -2 ≤ a ∧ a ≤ 2 → (∀ x : ℂ, x^2 + a*x + 1 = 0 → x.im ≠ 0)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_imaginary_roots_necessary_not_sufficient_l1347_134733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclone_pump_theorem_l1347_134762

/-- The Cyclone unit pumps water at a constant rate in gallons per hour. -/
noncomputable def pump_rate : ℝ := 500

/-- The time in hours for which we want to calculate the amount of water pumped. -/
noncomputable def time_in_hours : ℝ := 30 / 60

/-- The amount of water pumped is the product of the pump rate and the time. -/
noncomputable def water_pumped (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Theorem stating that the Cyclone unit pumps 250 gallons in 30 minutes. -/
theorem cyclone_pump_theorem : water_pumped pump_rate time_in_hours = 250 := by
  -- Unfold the definitions
  unfold water_pumped pump_rate time_in_hours
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclone_pump_theorem_l1347_134762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_tetrahedron_with_specific_squares_l1347_134725

/-- A tetrahedron with two perpendicular edges -/
structure Tetrahedron where
  edge1 : ℝ
  edge2 : ℝ
  perpendicular : True  -- We'll assume perpendicularity without proving it for now

/-- The side length of a square cross-section given two perpendicular edges -/
noncomputable def squareSideLength (t : Tetrahedron) (k : ℝ) : ℝ :=
  (k * t.edge1 * t.edge2) / (k * t.edge1 + t.edge2)

/-- Theorem stating the existence of a tetrahedron with specific square cross-sections -/
theorem exists_tetrahedron_with_specific_squares :
  ∃ t : Tetrahedron, ∃ k1 k2 : ℝ,
    squareSideLength t k1 = 100 ∧
    squareSideLength t k2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_tetrahedron_with_specific_squares_l1347_134725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_inverse_l1347_134724

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 4

noncomputable def g (x : ℝ) : ℝ := (2 : ℝ) ^ (2 * x)

-- State the theorem
theorem f_g_inverse : ∀ x > 0, f (g x) = x ∧ g (f x) = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_inverse_l1347_134724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l1347_134793

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ

-- Define the conditions
def angle_bisector_perpendicular (t : Triangle) : Prop :=
  let AB := (t.B.1 - t.A.1, t.B.2 - t.A.2)
  let AC := (t.C.1 - t.A.1, t.C.2 - t.A.2)
  let BC := (t.C.1 - t.B.1, t.C.2 - t.B.2)
  let AB_norm := Real.sqrt (AB.1^2 + AB.2^2)
  let AC_norm := Real.sqrt (AC.1^2 + AC.2^2)
  (AB.1 / AB_norm + AC.1 / AC_norm) * BC.1 +
  (AB.2 / AB_norm + AC.2 / AC_norm) * BC.2 = 0

def area_formula (t : Triangle) : Prop :=
  (t.a^2 + t.c^2 - t.b^2) / 4 = t.a * t.c * Real.sin t.angle_B / 2

-- Theorem statement
theorem triangle_shape (t : Triangle) 
  (h1 : angle_bisector_perpendicular t) 
  (h2 : area_formula t) : 
  t.angle_A = π/2 ∧ t.angle_B = π/4 ∧ t.angle_C = π/4 ∧ t.b = t.c :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l1347_134793
