import Mathlib

namespace max_value_of_f_l574_57403

open Real

noncomputable def f (x : ℝ) : ℝ := x / (exp x)

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x ∈ Set.Icc 0 2, f x ≤ f c) ∧
  f c = 1 / exp 1 := by
  sorry

end max_value_of_f_l574_57403


namespace cricket_game_run_rate_l574_57410

/-- Represents a cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  firstPartOvers : ℕ
  firstPartRunRate : ℚ
  target : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPartOvers
  let firstPartRuns := game.firstPartRunRate * game.firstPartOvers
  let remainingRuns := game.target - firstPartRuns
  remainingRuns / remainingOvers

/-- Theorem statement for the cricket game scenario -/
theorem cricket_game_run_rate 
  (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstPartOvers = 10)
  (h3 : game.firstPartRunRate = 3.2)
  (h4 : game.target = 282) :
  requiredRunRate game = 6.25 := by
  sorry

end cricket_game_run_rate_l574_57410


namespace difference_of_fractions_l574_57423

theorem difference_of_fractions : 
  (3 - 390 / 5) - (4 - 210 / 7) = -49 := by sorry

end difference_of_fractions_l574_57423


namespace square_minus_one_l574_57402

theorem square_minus_one (x : ℤ) (h : x^2 = 1521) : (x + 1) * (x - 1) = 1520 := by
  sorry

end square_minus_one_l574_57402


namespace solve_system_l574_57478

theorem solve_system (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : x * y = 2 * (x + y))
  (eq2 : y * z = 4 * (y + z))
  (eq3 : x * z = 8 * (x + z)) :
  x = 16 / 3 := by
sorry

end solve_system_l574_57478


namespace greatest_power_of_three_in_factorial_l574_57412

/-- The greatest exponent x such that 3^x divides 22! is 9 -/
theorem greatest_power_of_three_in_factorial : 
  (∃ x : ℕ, x = 9 ∧ 
    (∀ y : ℕ, 3^y ∣ Nat.factorial 22 → y ≤ x) ∧
    (3^x ∣ Nat.factorial 22)) := by sorry

end greatest_power_of_three_in_factorial_l574_57412


namespace cycle_loss_percentage_l574_57427

/-- Calculates the percentage of loss given the cost price and selling price. -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Proves that the loss percentage for a cycle with cost price 1400 and selling price 1190 is 15%. -/
theorem cycle_loss_percentage :
  let cost_price : ℚ := 1400
  let selling_price : ℚ := 1190
  loss_percentage cost_price selling_price = 15 := by
sorry

end cycle_loss_percentage_l574_57427


namespace largest_sum_and_simplification_l574_57448

theorem largest_sum_and_simplification : 
  let sums := [2/5 + 1/6, 2/5 + 1/3, 2/5 + 1/7, 2/5 + 1/8, 2/5 + 1/9]
  (∀ x ∈ sums, x ≤ 2/5 + 1/3) ∧ (2/5 + 1/3 = 11/15) := by
  sorry

end largest_sum_and_simplification_l574_57448


namespace parallel_line_through_point_l574_57431

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Checks if a point lies on a line -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- The given line 4x + 2y = 8 -/
def givenLine : Line :=
  { slope := -2, intercept := 4 }

/-- The line we need to prove -/
def parallelLine : Line :=
  { slope := -2, intercept := 1 }

theorem parallel_line_through_point :
  parallel parallelLine givenLine ∧
  pointOnLine parallelLine 0 1 :=
sorry

end parallel_line_through_point_l574_57431


namespace secret_organization_membership_l574_57454

theorem secret_organization_membership (total_cents : ℕ) (max_members : ℕ) : 
  total_cents = 300737 ∧ max_members = 500 →
  ∃! (members : ℕ) (fee_cents : ℕ),
    members ≤ max_members ∧
    members * fee_cents = total_cents ∧
    members = 311 ∧
    fee_cents = 967 := by
  sorry

end secret_organization_membership_l574_57454


namespace determinant_inequality_solution_l574_57492

-- Define the determinant of a 2x2 matrix
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the solution set
def solution_set : Set ℝ := {x | -1 < x ∧ x < 3}

-- Theorem statement
theorem determinant_inequality_solution :
  {x : ℝ | det 1 2 x (x^2) < 3} = solution_set :=
sorry

end determinant_inequality_solution_l574_57492


namespace octal_2016_to_binary_l574_57444

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ := sorry

/-- Converts a decimal number to binary --/
def decimal_to_binary (decimal : ℕ) : List ℕ := sorry

/-- Converts an octal number to binary --/
def octal_to_binary (octal : ℕ) : List ℕ :=
  decimal_to_binary (octal_to_decimal octal)

theorem octal_2016_to_binary :
  octal_to_binary 2016 = [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0] := by sorry

end octal_2016_to_binary_l574_57444


namespace average_sales_per_month_l574_57433

def sales_data : List ℕ := [100, 60, 40, 120]

theorem average_sales_per_month :
  (List.sum sales_data) / (List.length sales_data) = 80 := by
  sorry

end average_sales_per_month_l574_57433


namespace mower_team_size_l574_57469

/-- Represents the mowing rate of one mower per day -/
def mower_rate : ℝ := 1

/-- Represents the area of the smaller meadow -/
def small_meadow : ℝ := 2 * mower_rate

/-- Represents the area of the larger meadow -/
def large_meadow : ℝ := 2 * small_meadow

/-- Represents the number of mowers in the team -/
def team_size : ℕ := 8

theorem mower_team_size :
  (team_size : ℝ) * mower_rate / 2 + (team_size : ℝ) * mower_rate / 2 = large_meadow ∧
  (team_size : ℝ) * mower_rate / 4 + mower_rate = small_meadow :=
by sorry

#check mower_team_size

end mower_team_size_l574_57469


namespace subtract_like_terms_l574_57462

theorem subtract_like_terms (x y : ℝ) : 5 * x * y - 4 * x * y = x * y := by
  sorry

end subtract_like_terms_l574_57462


namespace no_integer_solution_for_2007_l574_57434

theorem no_integer_solution_for_2007 :
  ¬ ∃ (a b c : ℤ), a^2 + b^2 + c^2 = 2007 := by
  sorry

end no_integer_solution_for_2007_l574_57434


namespace log_equation_solution_l574_57496

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation_solution (m n b : ℝ) (h : lg m = b - lg n) : m = 10^b / n := by
  sorry

end log_equation_solution_l574_57496


namespace g_forms_l574_57486

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the property for g
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9*x^2 - 6*x + 1

-- Theorem statement
theorem g_forms {g : ℝ → ℝ} (h : g_property g) :
  (∀ x, g x = 3*x - 1) ∨ (∀ x, g x = -3*x + 1) := by
  sorry

end g_forms_l574_57486


namespace cos_beta_eq_four_fifths_l574_57474

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_E_eq_angle_G (q : Quadrilateral) (β : ℝ) : Prop := sorry

def side_EF_eq_side_GH (q : Quadrilateral) : Prop := sorry

def side_EH_ne_side_FG (q : Quadrilateral) : Prop := sorry

def perimeter (q : Quadrilateral) : ℝ := sorry

-- Main theorem
theorem cos_beta_eq_four_fifths (q : Quadrilateral) (β : ℝ) :
  is_convex q →
  angle_E_eq_angle_G q β →
  side_EF_eq_side_GH q →
  side_EH_ne_side_FG q →
  perimeter q = 720 →
  Real.cos β = 4/5 := by sorry

end cos_beta_eq_four_fifths_l574_57474


namespace sum_of_solutions_l574_57452

theorem sum_of_solutions (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ a b c : ℝ) 
  (eq1 : a₁ * (b₂ * c₃ - b₃ * c₂) - a₂ * (b₁ * c₃ - b₃ * c₁) + a₃ * (b₁ * c₂ - b₂ * c₁) = 9)
  (eq2 : a * (b₂ * c₃ - b₃ * c₂) - a₂ * (b * c₃ - b₃ * c) + a₃ * (b * c₂ - b₂ * c) = 17)
  (eq3 : a₁ * (b * c₃ - b₃ * c) - a * (b₁ * c₃ - b₃ * c₁) + a₃ * (b₁ * c - b * c₁) = -8)
  (eq4 : a₁ * (b₂ * c - b * c₂) - a₂ * (b₁ * c - b * c₁) + a * (b₁ * c₂ - b₂ * c₁) = 7)
  (sys1 : a₁ * x + a₂ * y + a₃ * z = a)
  (sys2 : b₁ * x + b₂ * y + b₃ * z = b)
  (sys3 : c₁ * x + c₂ * y + c₃ * z = c) :
  x + y + z = 16/9 := by
sorry

end sum_of_solutions_l574_57452


namespace smallest_integer_with_remainders_l574_57447

theorem smallest_integer_with_remainders : ∃ n : ℕ,
  n > 1 ∧
  n % 13 = 2 ∧
  n % 7 = 2 ∧
  n % 3 = 2 ∧
  ∀ m : ℕ, m > 1 → m % 13 = 2 → m % 7 = 2 → m % 3 = 2 → n ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_integer_with_remainders_l574_57447


namespace cubic_root_h_value_l574_57481

theorem cubic_root_h_value : ∀ h : ℚ, 
  (3 : ℚ)^3 + h * 3 - 20 = 0 → h = -7/3 := by
  sorry

end cubic_root_h_value_l574_57481


namespace area_gray_region_l574_57421

/-- The area of the gray region between two concentric circles -/
theorem area_gray_region (r : ℝ) (h1 : r > 0) (h2 : 2 * r = r + 3) : 
  π * (2 * r)^2 - π * r^2 = 27 * π := by
  sorry

end area_gray_region_l574_57421


namespace parallel_vectors_x_value_l574_57479

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (1, 6)
  parallel a b → x = 1/3 := by
  sorry

end parallel_vectors_x_value_l574_57479


namespace tangent_product_l574_57401

theorem tangent_product (A B : ℝ) (hA : A = 30 * π / 180) (hB : B = 60 * π / 180) :
  (1 + Real.tan A) * (1 + Real.tan B) = 2 + 4 * Real.sqrt 3 / 3 := by
  sorry

end tangent_product_l574_57401


namespace combination_sum_equals_4950_l574_57491

theorem combination_sum_equals_4950 : Nat.choose 99 98 + Nat.choose 99 97 = 4950 := by
  sorry

end combination_sum_equals_4950_l574_57491


namespace cos_sin_identity_l574_57465

theorem cos_sin_identity :
  Real.cos (40 * π / 180) * Real.cos (160 * π / 180) + Real.sin (40 * π / 180) * Real.sin (20 * π / 180) = -1/2 :=
by sorry

end cos_sin_identity_l574_57465


namespace fewer_girls_than_boys_l574_57464

theorem fewer_girls_than_boys (total_students : ℕ) (girls_ratio boys_ratio : ℕ) : 
  total_students = 24 →
  girls_ratio = 3 →
  boys_ratio = 5 →
  total_students * girls_ratio / (girls_ratio + boys_ratio) = 9 ∧
  total_students * boys_ratio / (girls_ratio + boys_ratio) = 15 ∧
  15 - 9 = 6 :=
by sorry

end fewer_girls_than_boys_l574_57464


namespace mississippi_arrangements_l574_57483

theorem mississippi_arrangements : 
  (11 : ℕ).factorial / ((4 : ℕ).factorial * (4 : ℕ).factorial * (2 : ℕ).factorial) = 34650 := by
  sorry

end mississippi_arrangements_l574_57483


namespace combined_work_time_l574_57498

/-- The time taken for three people to complete a task together, given their individual rates --/
theorem combined_work_time (rate_shawn rate_karen rate_alex : ℚ) 
  (h_shawn : rate_shawn = 1 / 18)
  (h_karen : rate_karen = 1 / 12)
  (h_alex : rate_alex = 1 / 15) :
  1 / (rate_shawn + rate_karen + rate_alex) = 180 / 37 :=
by sorry

end combined_work_time_l574_57498


namespace gcd_upper_bound_l574_57494

theorem gcd_upper_bound (a b c : ℕ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  Nat.gcd (a * b + 1) (Nat.gcd (a * c + 1) (b * c + 1)) ≤ (a + b + c) / 3 :=
sorry

end gcd_upper_bound_l574_57494


namespace cube_root_of_110592_l574_57409

theorem cube_root_of_110592 :
  ∃! (x : ℕ), x^3 = 110592 ∧ x > 0 :=
by
  use 48
  constructor
  · simp
  · intro y hy
    sorry

#eval 48^3  -- This will output 110592

end cube_root_of_110592_l574_57409


namespace book_arrangement_l574_57488

theorem book_arrangement (n m : ℕ) (h : n + m = 8) :
  Nat.choose 8 n = 56 :=
sorry

end book_arrangement_l574_57488


namespace exactly_one_number_satisfies_condition_l574_57428

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  n < 700 ∧ n = 7 * sum_of_digits n

theorem exactly_one_number_satisfies_condition : 
  ∃! n : ℕ, satisfies_condition n :=
sorry

end exactly_one_number_satisfies_condition_l574_57428


namespace membership_fee_increase_l574_57459

/-- Proves that the yearly increase in membership fee is $10 given the initial and final fees -/
theorem membership_fee_increase
  (initial_fee : ℕ)
  (final_fee : ℕ)
  (initial_year : ℕ)
  (final_year : ℕ)
  (h1 : initial_fee = 80)
  (h2 : final_fee = 130)
  (h3 : initial_year = 1)
  (h4 : final_year = 6)
  (h5 : final_fee = initial_fee + (final_year - initial_year) * (yearly_increase : ℕ)) :
  yearly_increase = 10 := by
  sorry

end membership_fee_increase_l574_57459


namespace complete_square_equation_l574_57414

theorem complete_square_equation : ∃ (a b c : ℤ), 
  (a > 0) ∧ 
  (∀ x : ℝ, 64 * x^2 + 80 * x - 81 = 0 ↔ (a * x + b)^2 = c) ∧
  (a = 8 ∧ b = 5 ∧ c = 106) := by
  sorry

end complete_square_equation_l574_57414


namespace inequality_holds_l574_57477

theorem inequality_holds (a : ℝ) (h : a ≥ 7/2) : 
  ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → 
    ∀ x : ℝ, (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + 
              (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8 := by
  sorry

end inequality_holds_l574_57477


namespace post_height_l574_57487

/-- Calculates the height of a cylindrical post given the squirrel's travel conditions -/
theorem post_height (total_distance : ℝ) (post_circumference : ℝ) (height_per_circuit : ℝ) : 
  total_distance = 27 ∧ post_circumference = 3 ∧ height_per_circuit = 3 →
  (total_distance / post_circumference) * height_per_circuit = 27 := by
sorry

end post_height_l574_57487


namespace complex_magnitude_l574_57471

theorem complex_magnitude (z : ℂ) (h : (1 + 2*I)*z = -3 + 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l574_57471


namespace apple_count_l574_57489

theorem apple_count (initial_oranges : ℕ) (removed_oranges : ℕ) (apples : ℕ) : 
  initial_oranges = 23 →
  removed_oranges = 13 →
  apples = (initial_oranges - removed_oranges) →
  apples = 10 :=
by
  sorry

end apple_count_l574_57489


namespace max_product_bound_l574_57436

/-- A three-digit number without zeros -/
structure ThreeDigitNoZero where
  a : ℕ
  b : ℕ
  c : ℕ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10

/-- The value of a three-digit number -/
def value (n : ThreeDigitNoZero) : ℕ :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of reciprocals of digits -/
def sum_reciprocals (n : ThreeDigitNoZero) : ℚ :=
  1 / n.a + 1 / n.b + 1 / n.c

/-- The product of the number and the sum of reciprocals of its digits -/
def product (n : ThreeDigitNoZero) : ℚ :=
  (value n : ℚ) * sum_reciprocals n

theorem max_product_bound :
  ∀ n : ThreeDigitNoZero, product n ≤ 1923.222 := by
  sorry

end max_product_bound_l574_57436


namespace campers_rowing_difference_l574_57476

theorem campers_rowing_difference (morning_campers afternoon_campers evening_campers : ℕ) 
  (h1 : morning_campers = 44)
  (h2 : afternoon_campers = 39)
  (h3 : evening_campers = 31) :
  morning_campers - afternoon_campers = 5 := by
  sorry

end campers_rowing_difference_l574_57476


namespace total_distance_walked_l574_57453

/-- The total distance walked by two girls, given one walked twice as far as the other -/
theorem total_distance_walked (nadia_distance : ℝ) (h_nadia : nadia_distance = 18) 
  (h_twice : nadia_distance = 2 * (nadia_distance / 2)) : 
  nadia_distance + (nadia_distance / 2) = 27 := by
  sorry

#check total_distance_walked

end total_distance_walked_l574_57453


namespace stating_pyramid_levels_for_1023_toothpicks_l574_57426

/-- Represents the number of toothpicks in a pyramid level. -/
def toothpicks_in_level (n : ℕ) : ℕ := 2^(n - 1)

/-- Represents the total number of toothpicks used up to a given level. -/
def total_toothpicks (n : ℕ) : ℕ := 2^n - 1

/-- 
Theorem stating that a pyramid with 1023 toothpicks has 10 levels,
where each level doubles the number of toothpicks from the previous level.
-/
theorem pyramid_levels_for_1023_toothpicks : 
  ∃ n : ℕ, n = 10 ∧ total_toothpicks n = 1023 := by
  sorry


end stating_pyramid_levels_for_1023_toothpicks_l574_57426


namespace hall_length_proof_l574_57449

/-- Proves that a hall with given dimensions and mat cost has a specific length -/
theorem hall_length_proof (width height mat_cost_per_sqm total_cost : ℝ) 
  (h_width : width = 15)
  (h_height : height = 5)
  (h_mat_cost : mat_cost_per_sqm = 40)
  (h_total_cost : total_cost = 38000) :
  ∃ (length : ℝ), 
    length = 32 ∧ 
    total_cost = mat_cost_per_sqm * (length * width + 2 * length * height + 2 * width * height) :=
by sorry

end hall_length_proof_l574_57449


namespace g_50_not_18_l574_57443

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- The smallest positive divisor of n -/
def smallest_divisor (n : ℕ+) : ℕ+ := sorry

/-- g₁ function as defined in the problem -/
def g₁ (n : ℕ+) : ℕ := (num_divisors n) * (smallest_divisor n).val

/-- General gⱼ function for j ≥ 1 -/
def g (j : ℕ) (n : ℕ+) : ℕ :=
  match j with
  | 0 => n.val
  | 1 => g₁ n
  | j+1 => g₁ ⟨g j n, sorry⟩

/-- Main theorem: For all positive integers n ≤ 100, g₅₀(n) ≠ 18 -/
theorem g_50_not_18 : ∀ n : ℕ+, n.val ≤ 100 → g 50 n ≠ 18 := by sorry

end g_50_not_18_l574_57443


namespace club_size_after_five_years_l574_57400

/-- Calculates the number of people in the club after a given number of years -/
def club_size (initial_size : ℕ) (years : ℕ) : ℕ :=
  match years with
  | 0 => initial_size
  | n + 1 => 4 * (club_size initial_size n - 7) + 7

theorem club_size_after_five_years :
  club_size 21 5 = 14343 := by
  sorry

#eval club_size 21 5

end club_size_after_five_years_l574_57400


namespace pq_passes_through_centroid_l574_57424

-- Define the points
variable (A B C D E F P Q : ℝ × ℝ)

-- Define the properties of the triangle and points
def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

def is_altitude_foot (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0

def is_centroid (E A C D : ℝ × ℝ) : Prop :=
  E.1 = (A.1 + C.1 + D.1) / 3 ∧ E.2 = (A.2 + C.2 + D.2) / 3

def is_perpendicular (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

def equal_distance (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

def line_passes_through_point (P Q X : ℝ × ℝ) : Prop :=
  (Q.2 - P.2) * (X.1 - P.1) = (Q.1 - P.1) * (X.2 - P.2)

def centroid (G A B C : ℝ × ℝ) : Prop :=
  G.1 = (A.1 + B.1 + C.1) / 3 ∧ G.2 = (A.2 + B.2 + C.2) / 3

-- State the theorem
theorem pq_passes_through_centroid
  (h1 : is_right_triangle A B C)
  (h2 : is_altitude_foot A B C D)
  (h3 : is_centroid E A C D)
  (h4 : is_centroid F B C D)
  (h5 : is_perpendicular C E P)
  (h6 : equal_distance C P A)
  (h7 : is_perpendicular C F Q)
  (h8 : equal_distance C Q B) :
  ∃ G, centroid G A B C ∧ line_passes_through_point P Q G :=
sorry

end pq_passes_through_centroid_l574_57424


namespace prime_factor_sum_l574_57493

theorem prime_factor_sum (w x y z k : ℕ) :
  2^w * 3^x * 5^y * 7^z * 11^k = 2520 →
  2*w + 3*x + 5*y + 7*z + 11*k = 24 := by
  sorry

end prime_factor_sum_l574_57493


namespace cat_whiskers_relationship_l574_57466

theorem cat_whiskers_relationship (princess_puff_whiskers catman_do_whiskers : ℕ) 
  (h1 : princess_puff_whiskers = 14) 
  (h2 : catman_do_whiskers = 22) : 
  (catman_do_whiskers - princess_puff_whiskers = 8) ∧ 
  (catman_do_whiskers : ℚ) / (princess_puff_whiskers : ℚ) = 11 / 7 := by
  sorry

end cat_whiskers_relationship_l574_57466


namespace trigonometric_system_solution_l574_57463

theorem trigonometric_system_solution (x y : ℝ) :
  (Real.sin x * Real.sin y = 0.75) →
  (Real.tan x * Real.tan y = 3) →
  ∃ (k n : ℤ), 
    (x = π/3 + π*(k + n : ℝ) ∨ x = -π/3 + π*(k + n : ℝ)) ∧
    (y = π/3 + π*(n - k : ℝ) ∨ y = -π/3 + π*(n - k : ℝ)) := by
  sorry

end trigonometric_system_solution_l574_57463


namespace slower_speed_percentage_l574_57451

theorem slower_speed_percentage (usual_time slower_time : ℝ) 
  (h1 : usual_time = 8)
  (h2 : slower_time = usual_time + 24) :
  (usual_time / slower_time) * 100 = 25 := by
sorry

end slower_speed_percentage_l574_57451


namespace f_is_direct_proportion_l574_57415

def f (x : ℝ) : ℝ := 3 * x

theorem f_is_direct_proportion : 
  (∀ x : ℝ, f x = 3 * x) ∧ 
  (f 0 = 0) ∧ 
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f x / x = f y / y) := by
  sorry

end f_is_direct_proportion_l574_57415


namespace deans_vacation_cost_l574_57455

/-- The total cost of a group vacation given the number of people and individual costs -/
def vacation_cost (num_people : ℕ) (rent transport food activities : ℚ) : ℚ :=
  num_people * (rent + transport + food + activities)

/-- Theorem stating the total cost for Dean's group vacation -/
theorem deans_vacation_cost :
  vacation_cost 7 70 25 55 40 = 1330 := by
  sorry

end deans_vacation_cost_l574_57455


namespace father_age_l574_57461

/-- Represents the ages of a family -/
structure FamilyAges where
  yy : ℕ
  cousin : ℕ
  mother : ℕ
  father : ℕ

/-- Defines the conditions of the problem -/
def problem_conditions (ages : FamilyAges) : Prop :=
  ages.yy = ages.cousin + 3 ∧
  ages.father = ages.mother + 4 ∧
  ages.yy + ages.cousin + ages.mother + ages.father = 95 ∧
  (ages.yy - 8) + (ages.cousin - 8) + (ages.mother - 8) + (ages.father - 8) = 65

/-- The theorem to be proved -/
theorem father_age (ages : FamilyAges) :
  problem_conditions ages → ages.father = 42 := by
  sorry

end father_age_l574_57461


namespace cosine_sine_identity_l574_57442

theorem cosine_sine_identity (α : Real) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6)^2 = -(Real.sqrt 3 + 2) / 3 := by
  sorry

end cosine_sine_identity_l574_57442


namespace inverse_proportion_ratio_l574_57419

-- Define the inverse proportionality relation
def inversely_proportional (a b : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, a x * b x = k

theorem inverse_proportion_ratio 
  (a b : ℝ → ℝ) (a₁ a₂ b₁ b₂ : ℝ) :
  inversely_proportional a b →
  a₁ ≠ 0 → a₂ ≠ 0 → b₁ ≠ 0 → b₂ ≠ 0 →
  a₁ / a₂ = 3 / 4 →
  b₁ - b₂ = 5 →
  b₁ / b₂ = 4 / 3 := by
    sorry

end inverse_proportion_ratio_l574_57419


namespace triangular_square_triangular_l574_57470

/-- Definition of triangular number -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Statement: 1 and 6 are the only triangular numbers whose squares are also triangular numbers -/
theorem triangular_square_triangular :
  ∀ n : ℕ, (∃ m : ℕ, (triangular n)^2 = triangular m) ↔ n = 1 ∨ n = 3 := by
sorry

end triangular_square_triangular_l574_57470


namespace area_difference_value_l574_57440

/-- A square with side length 2 units -/
def square_side : ℝ := 2

/-- Right-angled isosceles triangle with legs of length 2 -/
def large_triangle_leg : ℝ := 2

/-- Right-angled isosceles triangle with legs of length 1 -/
def small_triangle_leg : ℝ := 1

/-- The region R formed by the union of the square and all triangles -/
def R : Set (ℝ × ℝ) := sorry

/-- The smallest convex polygon S containing R -/
def S : Set (ℝ × ℝ) := sorry

/-- The area of the region inside S but outside R -/
def area_difference : ℝ := sorry

/-- Theorem stating the area difference between S and R -/
theorem area_difference_value : area_difference = (27 * Real.sqrt 3 - 28) / 2 := by sorry

end area_difference_value_l574_57440


namespace abc_inequality_l574_57425

theorem abc_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (h_abc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end abc_inequality_l574_57425


namespace five_three_number_properties_l574_57432

/-- Definition of a "five-three number" -/
def is_five_three_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 ∧ d ≥ 0 ∧ d ≤ 9 ∧
    a = c + 5 ∧ b = d + 3

/-- Definition of M(A) -/
def M (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a + c + 2 * (b + d)

/-- Definition of N(A) -/
def N (n : ℕ) : ℤ :=
  (n / 100 % 10) - 3

theorem five_three_number_properties :
  (∃ (max min : ℕ),
    is_five_three_number max ∧
    is_five_three_number min ∧
    (∀ n, is_five_three_number n → n ≤ max ∧ n ≥ min) ∧
    max - min = 4646) ∧
  (∃ A : ℕ,
    is_five_three_number A ∧
    (M A) % (N A) = 0 ∧
    A = 5401) :=
sorry

end five_three_number_properties_l574_57432


namespace july_birth_percentage_l574_57445

theorem july_birth_percentage (total_scientists : ℕ) (july_births : ℕ) : 
  total_scientists = 150 → july_births = 15 → 
  (july_births : ℚ) / (total_scientists : ℚ) * 100 = 10 := by
  sorry

end july_birth_percentage_l574_57445


namespace intersection_A_B_l574_57438

def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end intersection_A_B_l574_57438


namespace constant_term_expansion_l574_57441

theorem constant_term_expansion (x : ℝ) : 
  let expansion := (2*x - 1/(2*x))^6
  ∃ (a b c d e f g : ℝ), expansion = a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + f*x + (-20) :=
sorry

end constant_term_expansion_l574_57441


namespace cos_2theta_value_l574_57413

theorem cos_2theta_value (θ : ℝ) (h : Real.tan (θ + π/4) = (1/2) * Real.tan θ - 7/2) : 
  Real.cos (2 * θ) = -4/5 := by
  sorry

end cos_2theta_value_l574_57413


namespace modular_congruence_l574_57404

theorem modular_congruence (n : ℤ) : 
  0 ≤ n ∧ n < 103 ∧ (102 * n) % 103 = 74 % 103 → n % 103 = 29 % 103 := by
sorry

end modular_congruence_l574_57404


namespace savings_calculation_l574_57437

-- Define the income and ratio
def income : ℕ := 15000
def income_ratio : ℕ := 15
def expenditure_ratio : ℕ := 8

-- Define the function to calculate savings
def calculate_savings (inc : ℕ) (inc_ratio : ℕ) (exp_ratio : ℕ) : ℕ :=
  inc - (inc * exp_ratio) / inc_ratio

-- Theorem to prove
theorem savings_calculation :
  calculate_savings income income_ratio expenditure_ratio = 7000 := by
  sorry

end savings_calculation_l574_57437


namespace point_location_l574_57405

theorem point_location (x y : ℝ) (h1 : y > 3 * x) (h2 : y > 5 - 2 * x) :
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end point_location_l574_57405


namespace specific_sequence_common_difference_l574_57473

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  first_term : ℝ
  last_term : ℝ
  sum : ℝ
  is_arithmetic : ℝ → ℝ → ℝ → Prop

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ := 
  sorry

/-- Theorem stating the common difference of the specific sequence -/
theorem specific_sequence_common_difference :
  ∃ (seq : ArithmeticSequence), 
    seq.first_term = 5 ∧ 
    seq.last_term = 50 ∧ 
    seq.sum = 495 ∧ 
    common_difference seq = 45 / 17 := by
  sorry

end specific_sequence_common_difference_l574_57473


namespace function_value_at_two_l574_57495

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 0, prove that f(2) = -16 -/
theorem function_value_at_two (a b : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^5 + a*x^3 + b*x - 8
  f (-2) = 0 → f 2 = -16 := by
  sorry

end function_value_at_two_l574_57495


namespace simplify_and_evaluate_part1_simplify_and_evaluate_part2_l574_57446

-- Part 1
theorem simplify_and_evaluate_part1 :
  ∀ a : ℝ, 3*a*(a^2 - 2*a + 1) - 2*a^2*(a - 3) = a^3 + 3*a ∧
  3*2*(2^2 - 2*2 + 1) - 2*2^2*(2 - 3) = 14 :=
sorry

-- Part 2
theorem simplify_and_evaluate_part2 :
  ∀ x : ℝ, (x - 4)*(x - 2) - (x - 1)*(x + 3) = -8*x + 11 ∧
  ((-5/2) - 4)*((-5/2) - 2) - ((-5/2) - 1)*((-5/2) + 3) = 31 :=
sorry

end simplify_and_evaluate_part1_simplify_and_evaluate_part2_l574_57446


namespace waiter_tips_l574_57457

/-- Calculates the total tips earned by a waiter given the number of customers, 
    number of non-tipping customers, and the tip amount from each tipping customer. -/
def calculate_tips (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : ℕ :=
  (total_customers - non_tipping_customers) * tip_amount

/-- Theorem stating that under the given conditions, the waiter earns $32 in tips. -/
theorem waiter_tips : 
  calculate_tips 9 5 8 = 32 := by
  sorry

end waiter_tips_l574_57457


namespace min_cooking_time_is_15_l574_57420

/-- Represents the duration of each cooking step in minutes -/
structure CookingSteps :=
  (washPot : ℕ)
  (washVegetables : ℕ)
  (prepareNoodles : ℕ)
  (boilWater : ℕ)
  (cookNoodles : ℕ)

/-- Calculates the minimum cooking time given the cooking steps -/
def minCookingTime (steps : CookingSteps) : ℕ :=
  max steps.boilWater (steps.washPot + steps.washVegetables + steps.prepareNoodles + steps.cookNoodles)

/-- Theorem stating that the minimum cooking time for the given steps is 15 minutes -/
theorem min_cooking_time_is_15 (steps : CookingSteps) 
  (h1 : steps.washPot = 2)
  (h2 : steps.washVegetables = 6)
  (h3 : steps.prepareNoodles = 2)
  (h4 : steps.boilWater = 10)
  (h5 : steps.cookNoodles = 3) :
  minCookingTime steps = 15 := by
  sorry

end min_cooking_time_is_15_l574_57420


namespace cat_bowl_refill_days_l574_57497

theorem cat_bowl_refill_days (empty_bowl_weight : ℝ) (daily_food : ℝ) (weight_after_eating : ℝ) (eaten_amount : ℝ) :
  empty_bowl_weight = 420 →
  daily_food = 60 →
  weight_after_eating = 586 →
  eaten_amount = 14 →
  (weight_after_eating + eaten_amount - empty_bowl_weight) / daily_food = 3 := by
  sorry

end cat_bowl_refill_days_l574_57497


namespace parabola_coefficient_sum_l574_57422

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_coefficient_sum (p : Parabola) :
  p.x_coord (-4) = 5 →
  p.x_coord (-2) = 3 →
  p.a + p.b + p.c = -15/2 := by
  sorry

end parabola_coefficient_sum_l574_57422


namespace sally_eggs_l574_57458

-- Define what a dozen is
def dozen : ℕ := 12

-- Define the number of dozens Sally bought
def dozens_bought : ℕ := 4

-- Theorem: Sally bought 48 eggs
theorem sally_eggs : dozens_bought * dozen = 48 := by
  sorry

end sally_eggs_l574_57458


namespace inequality_solution_set_l574_57456

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) * Real.sqrt (x + 3) ≥ 0

-- Define the solution set
def solution_set : Set ℝ := {-3} ∪ Set.Ici 2

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set := by sorry

end inequality_solution_set_l574_57456


namespace imaginary_part_of_complex_fraction_l574_57416

theorem imaginary_part_of_complex_fraction :
  Complex.im ((1 + 2*Complex.I) / (1 + Complex.I)) = 1/2 := by
  sorry

end imaginary_part_of_complex_fraction_l574_57416


namespace expression_simplification_and_evaluation_l574_57480

theorem expression_simplification_and_evaluation :
  ∀ x : ℤ, -2 < x → x < 2 → x ≠ -1 → x ≠ 0 →
  (((x^2 - 1) / (x^2 + 2*x + 1)) / ((1 / (x + 1)) - 1) = (1 - x) / x) ∧
  (((x^2 - 1) / (x^2 + 2*x + 1)) / ((1 / (x + 1)) - 1) = 0) :=
by sorry

end expression_simplification_and_evaluation_l574_57480


namespace factory_employee_count_l574_57417

/-- Given a factory with three workshops and stratified sampling information, 
    prove the total number of employees. -/
theorem factory_employee_count 
  (x : ℕ) -- number of employees in Workshop A
  (y : ℕ) -- number of employees in Workshop C
  (h1 : x + 300 + y = 900) -- total employees
  (h2 : 20 + 15 + 10 = 45) -- stratified sample
  : x + 300 + y = 900 := by
  sorry

#check factory_employee_count

end factory_employee_count_l574_57417


namespace local_max_implies_c_eq_six_l574_57482

/-- Given a function f(x) = x(x-c)² where c is a constant, 
    if f has a local maximum at x = 2, then c = 6 -/
theorem local_max_implies_c_eq_six (c : ℝ) : 
  let f : ℝ → ℝ := λ x => x * (x - c)^2
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), f x ≤ f 2) →
  c = 6 := by
  sorry

end local_max_implies_c_eq_six_l574_57482


namespace equation_and_expression_proof_l574_57485

theorem equation_and_expression_proof :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 7 ∧ x2 = 2 - Real.sqrt 7 ∧
    x1^2 - 4*x1 - 3 = 0 ∧ x2^2 - 4*x2 - 3 = 0) ∧
  ((-1)^2 + 2 * Real.sin (π/3) - Real.tan (π/4) = Real.sqrt 3) :=
by sorry

end equation_and_expression_proof_l574_57485


namespace sum_of_specific_numbers_l574_57429

theorem sum_of_specific_numbers : 
  12345 + 23451 + 34512 + 45123 + 51234 = 166665 := by
  sorry

end sum_of_specific_numbers_l574_57429


namespace rigged_coin_probability_l574_57467

theorem rigged_coin_probability (p : ℝ) (h1 : p < 1/2) 
  (h2 : 20 * p^3 * (1-p)^3 = 1/12) : p = (1 - Real.sqrt 0.86) / 2 := by
  sorry

end rigged_coin_probability_l574_57467


namespace point_reflection_fourth_to_second_l574_57430

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem: If A(a,b) is in the fourth quadrant, then B(-b,-a) is in the second quadrant -/
theorem point_reflection_fourth_to_second (a b : ℝ) :
  is_in_fourth_quadrant (Point.mk a b) →
  is_in_second_quadrant (Point.mk (-b) (-a)) := by
  sorry


end point_reflection_fourth_to_second_l574_57430


namespace haji_mother_sales_l574_57475

theorem haji_mother_sales (tough_week_sales : ℕ) (good_weeks : ℕ) (tough_weeks : ℕ)
  (h1 : tough_week_sales = 800)
  (h2 : tough_week_sales * 2 = tough_week_sales + tough_week_sales)
  (h3 : good_weeks = 5)
  (h4 : tough_weeks = 3) :
  tough_week_sales * tough_weeks + (tough_week_sales * 2) * good_weeks = 10400 := by
  sorry

end haji_mother_sales_l574_57475


namespace quadratic_vertex_on_x_axis_l574_57450

/-- The quadratic function -x^2 + 4x + t has its vertex on the x-axis if and only if t = -4 -/
theorem quadratic_vertex_on_x_axis (t : ℝ) : 
  (∃ x : ℝ, ∀ y : ℝ, y = -x^2 + 4*x + t → y = 0) ↔ t = -4 := by
  sorry

end quadratic_vertex_on_x_axis_l574_57450


namespace line_through_midpoint_l574_57418

-- Define the points and lines
def P : ℝ × ℝ := (1, 2)
def L1 : ℝ → ℝ → Prop := λ x y => 3 * x - y + 2 = 0
def L2 : ℝ → ℝ → Prop := λ x y => x - 2 * y + 1 = 0

-- Define the property of A and B being on L1 and L2 respectively
def A_on_L1 (A : ℝ × ℝ) : Prop := L1 A.1 A.2
def B_on_L2 (B : ℝ × ℝ) : Prop := L2 B.1 B.2

-- Define the midpoint property
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0

-- State the theorem
theorem line_through_midpoint :
  ∀ (A B : ℝ × ℝ),
    A_on_L1 A →
    B_on_L2 B →
    is_midpoint P A B →
    ∀ (x y : ℝ),
      (∃ (t : ℝ), x = P.1 + t * (A.1 - P.1) ∧ y = P.2 + t * (A.2 - P.2)) →
      line_equation x y :=
sorry

end line_through_midpoint_l574_57418


namespace parabola_same_side_l574_57439

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by a quadratic function -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two points are on the same side of a parabola -/
def sameSide (p : Parabola) (A B : Point) : Prop :=
  (p.a * A.x^2 + p.b * A.x + p.c - A.y) * (p.a * B.x^2 + p.b * B.x + p.c - B.y) > 0

/-- The main theorem to prove -/
theorem parabola_same_side :
  let A : Point := ⟨-1, -1⟩
  let B : Point := ⟨0, 2⟩
  let p1 : Parabola := ⟨2, 4, 0⟩
  let p2 : Parabola := ⟨-1, 2, -1⟩
  let p3 : Parabola := ⟨-1, 0, 3⟩
  let p4 : Parabola := ⟨1/2, -1, -3/2⟩
  let p5 : Parabola := ⟨-1, -4, -3⟩
  sameSide p1 A B ∧ sameSide p2 A B ∧ sameSide p3 A B ∧
  ¬(sameSide p4 A B) ∧ ¬(sameSide p5 A B) :=
by sorry


end parabola_same_side_l574_57439


namespace average_of_other_results_l574_57484

theorem average_of_other_results
  (n₁ : ℕ) (n₂ : ℕ) (avg₁ : ℚ) (avg_all : ℚ)
  (h₁ : n₁ = 60)
  (h₂ : n₂ = 40)
  (h₃ : avg₁ = 40)
  (h₄ : avg_all = 48)
  : (n₁ * avg₁ + n₂ * ((n₁ + n₂) * avg_all - n₁ * avg₁) / n₂) / (n₁ + n₂) = avg_all ∧
    ((n₁ + n₂) * avg_all - n₁ * avg₁) / n₂ = 60 :=
by sorry

end average_of_other_results_l574_57484


namespace quadratic_distinct_roots_l574_57408

/-- The quadratic equation x^2 + 2x + m + 1 = 0 has two distinct real roots if and only if m < 0 -/
theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + m + 1 = 0 ∧ y^2 + 2*y + m + 1 = 0) ↔ m < 0 := by
  sorry

end quadratic_distinct_roots_l574_57408


namespace function_composition_ratio_l574_57435

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem function_composition_ratio :
  f (g (f 2)) / g (f (g 2)) = 41 / 7 := by
  sorry

end function_composition_ratio_l574_57435


namespace magnitude_of_b_is_one_l574_57460

/-- Given two vectors a and b in ℝ², prove that the magnitude of b is 1 -/
theorem magnitude_of_b_is_one (a b : ℝ × ℝ) : 
  (Real.cos (60 * π / 180) = a.fst * b.fst + a.snd * b.snd) →  -- angle between a and b is 60°
  (a.fst^2 + a.snd^2 = 1) →  -- |a| = 1
  ((2*a.fst - b.fst)^2 + (2*a.snd - b.snd)^2 = 3) →  -- |2a - b| = √3
  (b.fst^2 + b.snd^2 = 1) :=  -- |b| = 1
by sorry

end magnitude_of_b_is_one_l574_57460


namespace smallest_positive_solution_tan_equation_l574_57406

theorem smallest_positive_solution_tan_equation :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ (y : ℝ), y > 0 ∧ Real.tan (3 * y) - Real.tan (2 * y) = 1 / Real.cos (2 * y) → x ≤ y) ∧
  Real.tan (3 * x) - Real.tan (2 * x) = 1 / Real.cos (2 * x) ∧
  x = π / 6 := by
  sorry

end smallest_positive_solution_tan_equation_l574_57406


namespace battery_current_l574_57407

/-- Given a battery with voltage 48V, prove that when the resistance R is 12Ω, 
    the current I is 4A, where I is related to R by the function I = 48 / R. -/
theorem battery_current (R : ℝ) (I : ℝ) : 
  (I = 48 / R) → (R = 12) → (I = 4) := by
  sorry

end battery_current_l574_57407


namespace incorrect_expression_l574_57490

theorem incorrect_expression (x y : ℝ) (h : x / y = 3 / 4) : 
  (x - y) / y = -1 / 4 ∧ (x - y) / y ≠ 1 / 4 := by
  sorry

end incorrect_expression_l574_57490


namespace quadratic_roots_integer_P_l574_57472

theorem quadratic_roots_integer_P (P : ℤ) 
  (h1 : 5 < P) (h2 : P < 20) 
  (h3 : ∃ x y : ℤ, x^2 - 2*(2*P - 3)*x + 4*P^2 - 14*P + 8 = 0 ∧ 
                   y^2 - 2*(2*P - 3)*y + 4*P^2 - 14*P + 8 = 0 ∧ 
                   x ≠ y) : 
  P = 12 := by sorry

end quadratic_roots_integer_P_l574_57472


namespace average_string_length_l574_57499

theorem average_string_length (s1 s2 s3 : ℝ) (h1 : s1 = 1) (h2 : s2 = 3) (h3 : s3 = 5) :
  (s1 + s2 + s3) / 3 = 3 := by
  sorry

end average_string_length_l574_57499


namespace hyperbola_vertices_distance_l574_57468

/-- The distance between the vertices of the hyperbola x^2/48 - y^2/16 = 1 is 8√3 -/
theorem hyperbola_vertices_distance :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ x^2 / 48 - y^2 / 16
  ∃ (a b : ℝ), a ≠ b ∧ f (a, 0) = 1 ∧ f (b, 0) = 1 ∧ |a - b| = 8 * Real.sqrt 3 :=
by
  sorry


end hyperbola_vertices_distance_l574_57468


namespace smallest_root_of_g_l574_57411

def g (x : ℝ) : ℝ := 12 * x^4 - 8 * x^2 + 1

theorem smallest_root_of_g :
  let r := Real.sqrt (1/6)
  (g r = 0) ∧ (∀ x : ℝ, g x = 0 → x ≥ 0 → x ≥ r) :=
by sorry

end smallest_root_of_g_l574_57411
