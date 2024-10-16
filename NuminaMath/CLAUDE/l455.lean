import Mathlib

namespace NUMINAMATH_CALUDE_final_sum_after_operations_l455_45558

theorem final_sum_after_operations (x y T : ℝ) (h : x + y = T) :
  3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l455_45558


namespace NUMINAMATH_CALUDE_remainder_theorem_l455_45535

-- Define the polynomial f(r) = r^15 - 3
def f (r : ℝ) : ℝ := r^15 - 3

-- Theorem statement
theorem remainder_theorem (r : ℝ) : 
  (f r) % (r - 2) = 32765 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l455_45535


namespace NUMINAMATH_CALUDE_power_equation_l455_45550

theorem power_equation (a m n : ℝ) (hm : a^m = 3) (hn : a^n = 4) : a^(2*m + 3*n) = 576 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l455_45550


namespace NUMINAMATH_CALUDE_female_employees_with_advanced_degrees_l455_45562

theorem female_employees_with_advanced_degrees 
  (total_employees : ℕ) 
  (total_females : ℕ) 
  (total_advanced_degrees : ℕ) 
  (males_college_only : ℕ) 
  (h1 : total_employees = 160)
  (h2 : total_females = 90)
  (h3 : total_advanced_degrees = 80)
  (h4 : males_college_only = 40) :
  total_advanced_degrees - (total_employees - total_females - males_college_only) = 50 := by
  sorry

end NUMINAMATH_CALUDE_female_employees_with_advanced_degrees_l455_45562


namespace NUMINAMATH_CALUDE_tangent_line_at_one_monotonic_increase_intervals_l455_45500

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Theorem for the tangent line equation
theorem tangent_line_at_one :
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ x y : ℝ, y = f x → (x = 1 → a*x + b*y + c = 0)) ∧
  a = 2 ∧ b = -1 ∧ c = 1 := by sorry

-- Theorem for the intervals of monotonic increase
theorem monotonic_increase_intervals :
  ∃ (a : ℝ), a > 0 ∧
  (∀ x : ℝ, (x < -a ∨ x > a) → (∀ y : ℝ, x < y → f x < f y)) ∧
  a = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_monotonic_increase_intervals_l455_45500


namespace NUMINAMATH_CALUDE_reese_savings_problem_l455_45589

/-- Represents the percentage of savings spent in March -/
def march_spending_percentage : ℝ → Prop := λ M =>
  let initial_savings : ℝ := 11000
  let february_spending : ℝ := 0.2 * initial_savings
  let march_spending : ℝ := M * initial_savings
  let april_spending : ℝ := 1500
  let remaining : ℝ := 2900
  initial_savings - february_spending - march_spending - april_spending = remaining ∧
  M = 0.4

theorem reese_savings_problem :
  ∃ M : ℝ, march_spending_percentage M :=
sorry

end NUMINAMATH_CALUDE_reese_savings_problem_l455_45589


namespace NUMINAMATH_CALUDE_average_weight_l455_45573

/-- Given three weights a, b, and c, prove that their average is 45 kg
    under the following conditions:
    1. The average of a and b is 40 kg
    2. The average of b and c is 43 kg
    3. The weight of b is 31 kg -/
theorem average_weight (a b c : ℝ) 
  (avg_ab : (a + b) / 2 = 40)
  (avg_bc : (b + c) / 2 = 43)
  (weight_b : b = 31) :
  (a + b + c) / 3 = 45 := by
  sorry


end NUMINAMATH_CALUDE_average_weight_l455_45573


namespace NUMINAMATH_CALUDE_square_root_of_1_5625_l455_45530

theorem square_root_of_1_5625 : Real.sqrt 1.5625 = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1_5625_l455_45530


namespace NUMINAMATH_CALUDE_four_digit_square_palindromes_l455_45533

/-- A function that checks if a number is a 4-digit integer -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- A function that checks if a number is a palindrome -/
def is_palindrome (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n
  digits = digits.reverse

/-- The main theorem stating that there are exactly 3 numbers satisfying all conditions -/
theorem four_digit_square_palindromes :
  ∃! (s : Finset ℕ), s.card = 3 ∧ 
  (∀ n ∈ s, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n) ∧
  (∀ n, is_four_digit n → is_perfect_square n → is_palindrome n → n ∈ s) :=
sorry

end NUMINAMATH_CALUDE_four_digit_square_palindromes_l455_45533


namespace NUMINAMATH_CALUDE_star_value_of_a_l455_45569

-- Define the operation *
def star (a b : ℝ) : ℝ := 2 * a - b^3

-- Theorem statement
theorem star_value_of_a : 
  ∃ a : ℝ, star a 3 = 15 ∧ a = 21 :=
by sorry

end NUMINAMATH_CALUDE_star_value_of_a_l455_45569


namespace NUMINAMATH_CALUDE_weight_of_top_l455_45596

/-- Given 9 robots each weighing 0.8 kg and 7 tops with a total weight of 10.98 kg,
    the weight of one top is 0.54 kg. -/
theorem weight_of_top (robot_weight : ℝ) (total_weight : ℝ) (num_robots : ℕ) (num_tops : ℕ) :
  robot_weight = 0.8 →
  num_robots = 9 →
  num_tops = 7 →
  total_weight = 10.98 →
  total_weight = (↑num_robots * robot_weight) + (↑num_tops * 0.54) :=
by sorry

end NUMINAMATH_CALUDE_weight_of_top_l455_45596


namespace NUMINAMATH_CALUDE_johns_primary_colors_l455_45553

/-- Given that John has 5 liters of paint for each color and 15 liters of paint in total,
    prove that the number of primary colors he is using is 3. -/
theorem johns_primary_colors (paint_per_color : ℝ) (total_paint : ℝ) 
    (h1 : paint_per_color = 5)
    (h2 : total_paint = 15) :
    total_paint / paint_per_color = 3 := by
  sorry

end NUMINAMATH_CALUDE_johns_primary_colors_l455_45553


namespace NUMINAMATH_CALUDE_first_bell_weight_l455_45518

theorem first_bell_weight (w : ℝ) 
  (h1 : w > 0)  -- Ensuring positive weight
  (h2 : 2 * w > 0)  -- Weight of second bell
  (h3 : 4 * (2 * w) > 0)  -- Weight of third bell
  (h4 : w + 2 * w + 4 * (2 * w) = 550)  -- Total weight condition
  : w = 50 := by
  sorry

end NUMINAMATH_CALUDE_first_bell_weight_l455_45518


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l455_45585

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x ≤ 3}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl M ∩ N) = {x | x < -1 ∨ (2 < x ∧ x ≤ 3)} :=
sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l455_45585


namespace NUMINAMATH_CALUDE_token_game_result_l455_45537

def iterate_operation (n : ℕ) (f : ℤ → ℤ) (initial : ℤ) : ℤ :=
  match n with
  | 0 => initial
  | m + 1 => f (iterate_operation m f initial)

theorem token_game_result :
  let square (x : ℤ) := x * x
  let cube (x : ℤ) := x * x * x
  let iterations := 50
  let token1 := iterate_operation iterations square 2
  let token2 := iterate_operation iterations cube (-2)
  let token3 := iterate_operation iterations square 0
  token1 + token2 + token3 = -496 := by
  sorry

end NUMINAMATH_CALUDE_token_game_result_l455_45537


namespace NUMINAMATH_CALUDE_tennis_tournament_matches_l455_45541

theorem tennis_tournament_matches (n : ℕ) (b : ℕ) (h1 : n = 120) (h2 : b = 40) :
  let total_matches := n - 1
  total_matches = 119 ∧ total_matches % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tennis_tournament_matches_l455_45541


namespace NUMINAMATH_CALUDE_rational_function_value_at_two_l455_45559

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_linear : ∃ a b : ℝ, ∀ x, p x = a * x + b
  q_cubic : ∃ a b c d : ℝ, ∀ x, q x = a * x^3 + b * x^2 + c * x + d
  asymptote_neg_four : q (-4) = 0
  asymptote_one : q 1 = 0
  passes_origin : p 0 = 0 ∧ q 0 ≠ 0
  passes_neg_one_neg_two : p (-1) / q (-1) = -2

/-- The main theorem -/
theorem rational_function_value_at_two (f : RationalFunction) : f.p 2 / f.q 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_at_two_l455_45559


namespace NUMINAMATH_CALUDE_jerrys_shelf_l455_45520

/-- The number of books on Jerry's shelf -/
def num_books : ℕ := 3

/-- The number of action figures added later -/
def added_figures : ℕ := 2

/-- The difference between action figures and books after adding -/
def difference : ℕ := 3

/-- The initial number of action figures on Jerry's shelf -/
def initial_figures : ℕ := 4

theorem jerrys_shelf :
  initial_figures + added_figures = num_books + difference := by sorry

end NUMINAMATH_CALUDE_jerrys_shelf_l455_45520


namespace NUMINAMATH_CALUDE_elle_piano_practice_l455_45547

/-- The number of minutes Elle practices piano on a weekday -/
def weekday_practice : ℕ := 30

/-- The number of minutes Elle practices piano on Saturday -/
def saturday_practice : ℕ := 3 * weekday_practice

/-- The total number of minutes Elle practices piano in a week -/
def total_practice : ℕ := 5 * weekday_practice + saturday_practice

theorem elle_piano_practice :
  weekday_practice = 30 ∧
  saturday_practice = 3 * weekday_practice ∧
  total_practice = 5 * weekday_practice + saturday_practice ∧
  total_practice = 4 * 60 := by
  sorry

end NUMINAMATH_CALUDE_elle_piano_practice_l455_45547


namespace NUMINAMATH_CALUDE_shift_direct_proportion_l455_45522

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Represents a horizontal shift transformation on a function -/
def horizontalShift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  fun x => f (x - shift)

/-- The original direct proportion function y = -2x -/
def originalFunction : ℝ → ℝ :=
  fun x => -2 * x

theorem shift_direct_proportion :
  ∃ (f : LinearFunction),
    f.m = -2 ∧
    f.b = 6 ∧
    (∀ x, (horizontalShift originalFunction 3) x = f.m * x + f.b) := by
  sorry

end NUMINAMATH_CALUDE_shift_direct_proportion_l455_45522


namespace NUMINAMATH_CALUDE_a_range_l455_45502

/-- Given points A and B, if line AB is symmetric about x-axis and intersects a circle, then a is in [1/4, 2] -/
theorem a_range (a : ℝ) : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (a, 0)
  let line_AB_symmetric : Prop := ∃ (k : ℝ), ∀ (x y : ℝ), y = k * (x - a) ↔ -y = k * (x - (-2)) + 3
  let circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 2)^2 = 1}
  let line_AB_intersects_circle : Prop := ∃ (p : ℝ × ℝ), p ∈ circle ∧ ∃ (k : ℝ), p.2 - 0 = k * (p.1 - a)
  line_AB_symmetric → line_AB_intersects_circle → a ∈ Set.Icc (1/4 : ℝ) 2 :=
by
  sorry

end NUMINAMATH_CALUDE_a_range_l455_45502


namespace NUMINAMATH_CALUDE_inequality_solution_range_l455_45579

theorem inequality_solution_range (m : ℚ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ 
   (∀ x ∈ s, x < 0 ∧ (x - 1) / 2 + 3 > (x + m) / 3) ∧
   (∀ x : ℤ, x < 0 → (x - 1) / 2 + 3 > (x + m) / 3 → x ∈ s)) ↔ 
  (11 / 2 : ℚ) ≤ m ∧ m < 6 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l455_45579


namespace NUMINAMATH_CALUDE_system_solution_and_range_l455_45599

theorem system_solution_and_range (a x y : ℝ) : 
  (2 * x + y = 5 * a ∧ x - 3 * y = -a + 7) →
  (x = 2 * a + 1 ∧ y = a - 2) ∧
  (x ≥ 0 ∧ y < 0 ↔ -1/2 ≤ a ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_and_range_l455_45599


namespace NUMINAMATH_CALUDE_box_volume_l455_45524

theorem box_volume (sheet_length sheet_width cut_length : ℝ) 
  (h1 : sheet_length = 46)
  (h2 : sheet_width = 36)
  (h3 : cut_length = 8) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 4800 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l455_45524


namespace NUMINAMATH_CALUDE_sin_max_implies_even_l455_45517

theorem sin_max_implies_even (f : ℝ → ℝ) (φ a : ℝ) 
  (h1 : ∀ x, f x = Real.sin (2 * x + φ))
  (h2 : ∀ x, f x ≤ f a) :
  ∀ x, f (x + a) = f (-x + a) := by
sorry

end NUMINAMATH_CALUDE_sin_max_implies_even_l455_45517


namespace NUMINAMATH_CALUDE_weight_problem_l455_45563

/-- The weight problem -/
theorem weight_problem (student_weight sister_weight brother_weight : ℝ) : 
  (student_weight - 8 = sister_weight + brother_weight) →
  (brother_weight = sister_weight + 5) →
  (sister_weight + brother_weight = 180) →
  (student_weight = 188) :=
by
  sorry

end NUMINAMATH_CALUDE_weight_problem_l455_45563


namespace NUMINAMATH_CALUDE_log_equation_implies_y_value_l455_45510

-- Define a positive real number type for the base of logarithms
def PositiveReal := {x : ℝ | x > 0}

-- Define the logarithm function
noncomputable def log (base : PositiveReal) (x : PositiveReal) : ℝ := Real.log x / Real.log base.val

-- The main theorem
theorem log_equation_implies_y_value 
  (a b c x : PositiveReal) 
  (p q r y : ℝ) 
  (base : PositiveReal)
  (h1 : log base a / p = log base b / q)
  (h2 : log base b / q = log base c / r)
  (h3 : log base c / r = log base x)
  (h4 : x.val ≠ 1)
  (h5 : b.val^2 / (a.val * c.val) = x.val^y) :
  y = 2*q - p - r := by
  sorry

end NUMINAMATH_CALUDE_log_equation_implies_y_value_l455_45510


namespace NUMINAMATH_CALUDE_hearty_beads_count_l455_45532

/-- The number of packages of blue beads Hearty bought -/
def blue_packages : ℕ := 3

/-- The number of packages of red beads Hearty bought -/
def red_packages : ℕ := 5

/-- The number of beads in each package -/
def beads_per_package : ℕ := 40

/-- The total number of beads Hearty has -/
def total_beads : ℕ := (blue_packages + red_packages) * beads_per_package

theorem hearty_beads_count : total_beads = 320 := by
  sorry

end NUMINAMATH_CALUDE_hearty_beads_count_l455_45532


namespace NUMINAMATH_CALUDE_previous_height_l455_45501

theorem previous_height (current_height : ℝ) (growth_rate : ℝ) : 
  current_height = 126 ∧ growth_rate = 0.05 → 
  current_height / (1 + growth_rate) = 120 := by
  sorry

end NUMINAMATH_CALUDE_previous_height_l455_45501


namespace NUMINAMATH_CALUDE_girls_in_class_l455_45584

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (girls : ℕ) :
  total = 35 →
  ratio_girls = 3 →
  ratio_boys = 4 →
  girls * ratio_boys = (total - girls) * ratio_girls →
  girls = 15 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l455_45584


namespace NUMINAMATH_CALUDE_petya_marking_strategy_l455_45586

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a rectangle that can be placed on the board -/
structure Rectangle :=
  (length : ℕ)
  (width : ℕ)

/-- The minimum number of cells needed to be marked to uniquely determine 
    the position of a rectangle on a board -/
def min_marked_cells (b : Board) (r : Rectangle) : ℕ := sorry

/-- The main theorem stating the minimum number of cells Petya needs to mark -/
theorem petya_marking_strategy (b : Board) (r : Rectangle) : 
  b.rows = 13 ∧ b.cols = 13 ∧ r.length = 6 ∧ r.width = 1 →
  min_marked_cells b r = 84 := by sorry

end NUMINAMATH_CALUDE_petya_marking_strategy_l455_45586


namespace NUMINAMATH_CALUDE_remainder_5_2024_mod_17_l455_45568

theorem remainder_5_2024_mod_17 : 5^2024 % 17 = 16 := by sorry

end NUMINAMATH_CALUDE_remainder_5_2024_mod_17_l455_45568


namespace NUMINAMATH_CALUDE_snail_reaches_tree_in_26_days_l455_45555

/-- The number of days it takes for a snail to reach a tree given its daily movement pattern -/
def snail_journey_days (s l₁ l₂ : ℕ) : ℕ :=
  let daily_progress := l₁ - l₂
  let days_to_reach_near := (s - l₁) / daily_progress
  days_to_reach_near + 1

/-- Theorem stating that the snail reaches the tree in 26 days under the given conditions -/
theorem snail_reaches_tree_in_26_days :
  snail_journey_days 30 5 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_snail_reaches_tree_in_26_days_l455_45555


namespace NUMINAMATH_CALUDE_class_average_weight_l455_45507

theorem class_average_weight (n₁ : ℕ) (n₂ : ℕ) (w₁ : ℝ) (w_total : ℝ) :
  n₁ = 16 →
  n₂ = 8 →
  w₁ = 50.25 →
  w_total = 48.55 →
  ((n₁ * w₁ + n₂ * ((n₁ + n₂) * w_total - n₁ * w₁) / n₂) / (n₁ + n₂) = w_total) →
  ((n₁ + n₂) * w_total - n₁ * w₁) / n₂ = 45.15 :=
by sorry

end NUMINAMATH_CALUDE_class_average_weight_l455_45507


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l455_45515

theorem functional_equation_solutions (f : ℕ → ℕ) 
  (h : ∀ n m : ℕ, f (3 * n + 2 * m) = f n * f m) : 
  (∀ n, f n = 0) ∨ 
  (∀ n, f n = 1) ∨ 
  ((∀ n, n ≠ 0 → f n = 0) ∧ f 0 = 1) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l455_45515


namespace NUMINAMATH_CALUDE_halfway_between_one_fourth_and_one_seventh_l455_45583

theorem halfway_between_one_fourth_and_one_seventh :
  let x : ℚ := 11 / 56
  (x - 1 / 4 : ℚ) = (1 / 7 - x : ℚ) ∧ 
  x = (1 / 4 + 1 / 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_fourth_and_one_seventh_l455_45583


namespace NUMINAMATH_CALUDE_abcd_mod_11_l455_45529

theorem abcd_mod_11 (a b c d : ℕ) : 
  a < 11 → b < 11 → c < 11 → d < 11 →
  (a + 3*b + 4*c + 2*d) % 11 = 3 →
  (3*a + b + 2*c + d) % 11 = 5 →
  (2*a + 4*b + c + 3*d) % 11 = 7 →
  (a + b + c + d) % 11 = 2 →
  (a * b * c * d) % 11 = 9 := by
sorry

end NUMINAMATH_CALUDE_abcd_mod_11_l455_45529


namespace NUMINAMATH_CALUDE_negation_equivalence_l455_45505

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 4*x + 5 ≤ 0) ↔ (∀ x : ℝ, x^2 + 4*x + 5 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l455_45505


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l455_45528

/-- Given a function f(x) = ax^2 / (x+1), prove that if the slope of the tangent line
    at the point (1, f(1)) is 1, then a = 4/3 -/
theorem tangent_slope_implies_a (a : ℝ) :
  let f := fun x : ℝ => (a * x^2) / (x + 1)
  let f' := fun x : ℝ => ((a * x^2 + 2 * a * x) / (x + 1)^2)
  f' 1 = 1 → a = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l455_45528


namespace NUMINAMATH_CALUDE_f_at_negative_one_l455_45506

-- Define the polynomials g and f
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 2*x + 15
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 150*x + c

-- State the theorem
theorem f_at_negative_one (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0 ∧
    f b c r₁ = 0 ∧ f b c r₂ = 0 ∧ f b c r₃ = 0) →
  f b c (-1) = 3733.25 := by
sorry


end NUMINAMATH_CALUDE_f_at_negative_one_l455_45506


namespace NUMINAMATH_CALUDE_geometry_propositions_l455_45565

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)

-- Axioms for the properties of parallel and perpendicular
axiom parallel_transitive {a b c : Plane} : parallel a b → parallel a c → parallel b c
axiom perpendicular_from_line {l : Line} {a b : Plane} : 
  line_perpendicular_plane l a → line_parallel_plane l b → perpendicular a b

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- State the theorem
theorem geometry_propositions :
  -- Proposition ①
  (∀ a b c : Plane, parallel a b → parallel a c → parallel b c) ∧
  -- Proposition ③
  (∀ l : Line, ∀ a b : Plane, line_perpendicular_plane l a → line_parallel_plane l b → perpendicular a b) ∧
  -- Negation of Proposition ②
  ¬(∀ l : Line, ∀ a b : Plane, perpendicular a b → line_parallel_plane l a → line_perpendicular_plane l b) ∧
  -- Negation of Proposition ④
  ¬(∀ l1 l2 : Line, ∀ a : Plane, line_parallel l1 l2 → line_in_plane l2 a → line_parallel_plane l1 a) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l455_45565


namespace NUMINAMATH_CALUDE_B_power_200_is_identity_l455_45549

def B : Matrix (Fin 4) (Fin 4) ℝ := !![0,0,0,1; 1,0,0,0; 0,1,0,0; 0,0,1,0]

theorem B_power_200_is_identity :
  B ^ 200 = (1 : Matrix (Fin 4) (Fin 4) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_B_power_200_is_identity_l455_45549


namespace NUMINAMATH_CALUDE_calculation_proof_l455_45540

theorem calculation_proof : 
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l455_45540


namespace NUMINAMATH_CALUDE_trapezoid_area_is_42_l455_45593

/-- A trapezoid bounded by the lines y = x + 2, y = 12, y = 6, and x = 0 -/
structure Trapezoid where
  /-- The line y = x + 2 -/
  line1 : ℝ → ℝ := λ x => x + 2
  /-- The line y = 12 -/
  line2 : ℝ → ℝ := λ _ => 12
  /-- The line y = 6 -/
  line3 : ℝ → ℝ := λ _ => 6
  /-- The y-axis (x = 0) -/
  line4 : ℝ → ℝ := λ _ => 0

/-- The area of the trapezoid -/
def trapezoidArea (t : Trapezoid) : ℝ := 42

/-- Theorem stating that the area of the defined trapezoid is 42 square units -/
theorem trapezoid_area_is_42 (t : Trapezoid) : trapezoidArea t = 42 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_42_l455_45593


namespace NUMINAMATH_CALUDE_tourist_guide_distribution_l455_45571

/-- The number of ways to distribute n tourists among k guides, 
    where each guide must have at least one tourist -/
def validDistributions (n k : ℕ) : ℕ :=
  k^n - (k.choose 1) * (k-1)^n + (k.choose 2) * (k-2)^n

theorem tourist_guide_distribution :
  validDistributions 8 3 = 5796 := by
  sorry

end NUMINAMATH_CALUDE_tourist_guide_distribution_l455_45571


namespace NUMINAMATH_CALUDE_new_person_weight_l455_45545

/-- Given 8 persons, if replacing one person weighing 50 kg with a new person 
    increases the average weight by 2.5 kg, then the weight of the new person is 70 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 50 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * weight_increase + replaced_weight ∧
    new_weight = 70 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l455_45545


namespace NUMINAMATH_CALUDE_school_c_variance_l455_45538

/-- Represents the data for a school's strong math foundation group -/
structure SchoolData where
  students : ℕ
  average : ℝ
  variance : ℝ

/-- Represents the overall data for all schools -/
structure OverallData where
  total_students : ℕ
  average : ℝ
  variance : ℝ

/-- Theorem stating that given the conditions, the variance of school C is 12 -/
theorem school_c_variance
  (ratio : Fin 3 → ℕ)
  (h_ratio : ratio = ![3, 2, 1])
  (overall : OverallData)
  (h_overall : overall = { total_students := 48, average := 117, variance := 21.5 })
  (school_a : SchoolData)
  (h_school_a : school_a = { students := 24, average := 118, variance := 15 })
  (school_b : SchoolData)
  (h_school_b : school_b = { students := 16, average := 114, variance := 21 })
  (school_c : SchoolData)
  (h_school_c_students : school_c.students = 8) :
  school_c.variance = 12 := by
  sorry

end NUMINAMATH_CALUDE_school_c_variance_l455_45538


namespace NUMINAMATH_CALUDE_g_value_at_2_l455_45587

def g (x : ℝ) : ℝ := 3 * x^8 - 4 * x^4 + 2 * x^2 - 6

theorem g_value_at_2 (h : g (-2) = 10) : g 2 = 1402 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_2_l455_45587


namespace NUMINAMATH_CALUDE_not_right_triangle_sides_l455_45594

theorem not_right_triangle_sides (a b c : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 4) (h3 : c = Real.sqrt 5) :
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_sides_l455_45594


namespace NUMINAMATH_CALUDE_polynomial_simplification_l455_45527

theorem polynomial_simplification (x : ℝ) : 
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 - 4*x^3 + 6*x^3 
  = 4*x^3 - x^2 + 23*x - 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l455_45527


namespace NUMINAMATH_CALUDE_jade_handled_80_transactions_l455_45523

/-- Calculates the number of transactions Jade handled given the conditions of the problem. -/
def jade_transactions (mabel_transactions : ℕ) : ℕ :=
  let anthony_transactions := mabel_transactions + mabel_transactions / 10
  let cal_transactions := anthony_transactions * 2 / 3
  cal_transactions + 14

/-- Theorem stating that Jade handled 80 transactions given the conditions of the problem. -/
theorem jade_handled_80_transactions : jade_transactions 90 = 80 := by
  sorry

end NUMINAMATH_CALUDE_jade_handled_80_transactions_l455_45523


namespace NUMINAMATH_CALUDE_fraction_problem_l455_45572

theorem fraction_problem : 
  let number : ℝ := 14.500000000000002
  let result : ℝ := 126.15
  let fraction : ℝ := result / (number ^ 2)
  fraction = 0.6 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l455_45572


namespace NUMINAMATH_CALUDE_parabola_and_tangent_line_l455_45574

/-- Parabola with vertex at origin and focus on positive y-axis -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  h_focus_pos : focus.2 > 0
  h_focus_eq : focus = (0, p)

/-- Line through focus intersecting parabola -/
structure IntersectingLine (para : Parabola) where
  a : ℝ × ℝ
  b : ℝ × ℝ
  h_on_parabola_a : a.1^2 = 2 * para.p * a.2
  h_on_parabola_b : b.1^2 = 2 * para.p * b.2
  h_through_focus : ∃ t : ℝ, (1 - t) • a + t • b = para.focus

/-- Line with y-intercept 6 intersecting parabola -/
structure TangentLine (para : Parabola) where
  m : ℝ
  p : ℝ × ℝ
  q : ℝ × ℝ
  r : ℝ × ℝ
  h_on_parabola_p : p.1^2 = 2 * para.p * p.2
  h_on_parabola_q : q.1^2 = 2 * para.p * q.2
  h_on_line_p : p.2 = m * p.1 + 6
  h_on_line_q : q.2 = m * q.1 + 6
  h_r_on_directrix : r.2 = -para.p
  h_qfr_collinear : ∃ t : ℝ, (1 - t) • q + t • r = para.focus
  h_pr_tangent : (p.2 - r.2) / (p.1 - r.1) = p.1 / (2 * para.p)

theorem parabola_and_tangent_line (para : Parabola) 
  (line : IntersectingLine para) 
  (tline : TangentLine para) :
  (∀ (t : ℝ), (1 - t) • line.a + t • line.b - (0, 3) = (0, 1)) →
  (‖line.a - line.b‖ = 8) →
  (para.p = 2 ∧ (tline.m = 1/2 ∨ tline.m = -1/2)) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_tangent_line_l455_45574


namespace NUMINAMATH_CALUDE_sum_of_inverse_G_power_three_l455_45595

def G : ℕ → ℚ
  | 0 => 0
  | 1 => 8/3
  | (n+2) => 3 * G (n+1) - (1/2) * G n

theorem sum_of_inverse_G_power_three : ∑' n, 1 / G (3^n) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_inverse_G_power_three_l455_45595


namespace NUMINAMATH_CALUDE_number_and_square_difference_l455_45544

theorem number_and_square_difference (N : ℝ) : N^2 - N = 12 ↔ N = 4 ∨ N = -3 := by
  sorry

end NUMINAMATH_CALUDE_number_and_square_difference_l455_45544


namespace NUMINAMATH_CALUDE_factor_sum_l455_45552

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 4) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 15 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l455_45552


namespace NUMINAMATH_CALUDE_numerator_of_x_l455_45509

theorem numerator_of_x (x y a : ℝ) : 
  x + y = -10 → 
  x^2 + y^2 = 50 → 
  x = a / y → 
  a = 25 := by sorry

end NUMINAMATH_CALUDE_numerator_of_x_l455_45509


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l455_45531

theorem updated_mean_after_decrement (n : ℕ) (original_mean decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 47 →
  (n * original_mean - n * decrement) / n = 153 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l455_45531


namespace NUMINAMATH_CALUDE_range_of_p_l455_45597

-- Define the set A
def A (p : ℝ) : Set ℝ := {x | x^2 + (p+2)*x + 1 = 0}

-- Theorem statement
theorem range_of_p (p : ℝ) : (A p ∩ Set.Ioi 0 = ∅) → p > -4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l455_45597


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l455_45525

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    (P = 5/2 ∧ Q = 0 ∧ R = -5) ∧
    ∀ (x : ℚ), x ≠ 4 ∧ x ≠ 2 →
      5*x / ((x - 4) * (x - 2)^3) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^3 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l455_45525


namespace NUMINAMATH_CALUDE_reflection_matrix_values_l455_45580

theorem reflection_matrix_values (a b : ℚ) :
  let R : Matrix (Fin 2) (Fin 2) ℚ := !![a, 9/26; b, 17/26]
  (R * R = 1) → (a = -17/26 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_reflection_matrix_values_l455_45580


namespace NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l455_45536

def monday_fabric : ℕ := 20
def fabric_cost : ℕ := 2
def wednesday_ratio : ℚ := 1/4
def total_earnings : ℕ := 140

theorem tuesday_to_monday_ratio :
  ∃ (tuesday_fabric : ℕ),
    (monday_fabric * fabric_cost + 
     tuesday_fabric * fabric_cost + 
     (wednesday_ratio * tuesday_fabric) * fabric_cost = total_earnings) ∧
    (tuesday_fabric = monday_fabric) := by
  sorry

end NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l455_45536


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l455_45534

theorem algebraic_expression_equality (x : ℝ) : 
  x^2 + 3*x + 8 = 7 → 3*x^2 + 9*x - 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l455_45534


namespace NUMINAMATH_CALUDE_equation_equality_l455_45543

theorem equation_equality (a b : ℝ) : -a*b + 3*b*a = 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l455_45543


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l455_45576

theorem cubic_sum_theorem (p q r : ℝ) (hp : p ≠ q) (hq : q ≠ r) (hr : r ≠ p)
  (h : (p^3 + 8) / p = (q^3 + 8) / q ∧ (q^3 + 8) / q = (r^3 + 8) / r) :
  p^3 + q^3 + r^3 = -24 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l455_45576


namespace NUMINAMATH_CALUDE_yoongi_has_bigger_number_l455_45513

-- Define Yoongi's number
def yoongi_number : ℕ := 4

-- Define Jungkook's number
def jungkook_number : ℕ := 6 / 3

-- Theorem to prove
theorem yoongi_has_bigger_number : yoongi_number > jungkook_number := by
  sorry

end NUMINAMATH_CALUDE_yoongi_has_bigger_number_l455_45513


namespace NUMINAMATH_CALUDE_task_fraction_by_B_l455_45521

theorem task_fraction_by_B (a b : ℚ) : 
  (a = (2/5) * b) → (b = (5/7) * (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_task_fraction_by_B_l455_45521


namespace NUMINAMATH_CALUDE_cupcakes_remaining_l455_45560

/-- The number of cupcakes in a dozen -/
def dozen : ℕ := 12

/-- The number of cupcakes Dani brings -/
def cupcakes_brought : ℕ := (10 * dozen) + (dozen / 2)

/-- The total number of students in the class -/
def total_students : ℕ := 48

/-- The number of teachers -/
def teachers : ℕ := 2

/-- The number of teacher's aids -/
def teacher_aids : ℕ := 2

/-- The number of absent students -/
def absent_students : ℕ := 6

/-- The number of students on a field trip -/
def field_trip_students : ℕ := 8

/-- The number of people present in the class -/
def people_present : ℕ := total_students - absent_students - field_trip_students + teachers + teacher_aids

/-- The number of cupcakes left after distribution -/
def cupcakes_left : ℕ := cupcakes_brought - people_present

theorem cupcakes_remaining :
  cupcakes_left = 85 :=
sorry

end NUMINAMATH_CALUDE_cupcakes_remaining_l455_45560


namespace NUMINAMATH_CALUDE_tea_trader_profit_percentage_l455_45516

/-- Calculates the profit percentage for a tea trader --/
theorem tea_trader_profit_percentage
  (tea1_weight : ℝ)
  (tea1_cost : ℝ)
  (tea2_weight : ℝ)
  (tea2_cost : ℝ)
  (sale_price : ℝ)
  (h1 : tea1_weight = 80)
  (h2 : tea1_cost = 15)
  (h3 : tea2_weight = 20)
  (h4 : tea2_cost = 20)
  (h5 : sale_price = 20) :
  let total_cost := tea1_weight * tea1_cost + tea2_weight * tea2_cost
  let total_weight := tea1_weight + tea2_weight
  let cost_per_kg := total_cost / total_weight
  let profit_per_kg := sale_price - cost_per_kg
  let profit_percentage := (profit_per_kg / cost_per_kg) * 100
  profit_percentage = 25 := by
sorry


end NUMINAMATH_CALUDE_tea_trader_profit_percentage_l455_45516


namespace NUMINAMATH_CALUDE_district3_to_district1_ratio_l455_45566

/-- The number of voters in District 1 -/
def district1_voters : ℕ := 322

/-- The difference in voters between District 3 and District 2 -/
def district3_2_diff : ℕ := 19

/-- The total number of voters in all three districts -/
def total_voters : ℕ := 1591

/-- The ratio of voters in District 3 to District 1 -/
def voter_ratio : ℚ := 2

theorem district3_to_district1_ratio :
  ∃ (district2_voters district3_voters : ℕ),
    district2_voters = district3_voters - district3_2_diff ∧
    district1_voters + district2_voters + district3_voters = total_voters ∧
    district3_voters = (voter_ratio : ℚ) * district1_voters := by
  sorry

end NUMINAMATH_CALUDE_district3_to_district1_ratio_l455_45566


namespace NUMINAMATH_CALUDE_school_boys_count_l455_45592

theorem school_boys_count (total_pupils : ℕ) (girls : ℕ) (boys : ℕ) : 
  total_pupils = 485 → girls = 232 → boys = total_pupils - girls → boys = 253 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l455_45592


namespace NUMINAMATH_CALUDE_log_sum_equals_one_l455_45551

theorem log_sum_equals_one : Real.log 2 + 2 * Real.log (Real.sqrt 5) = Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_one_l455_45551


namespace NUMINAMATH_CALUDE_avery_egg_cartons_l455_45575

/-- The number of egg cartons that can be filled given the number of chickens,
    eggs per chicken, and eggs per carton. -/
def egg_cartons_filled (num_chickens : ℕ) (eggs_per_chicken : ℕ) (eggs_per_carton : ℕ) : ℕ :=
  (num_chickens * eggs_per_chicken) / eggs_per_carton

/-- Theorem stating that with 20 chickens, each laying 6 eggs, and egg cartons
    that hold 12 eggs each, the number of egg cartons that can be filled is 10. -/
theorem avery_egg_cartons :
  egg_cartons_filled 20 6 12 = 10 := by
  sorry

end NUMINAMATH_CALUDE_avery_egg_cartons_l455_45575


namespace NUMINAMATH_CALUDE_equal_segments_after_rearrangement_l455_45564

-- Define a line in a plane
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a right-angled triangle
structure RightTriangle :=
  (leg1 : ℝ)
  (leg2 : ℝ)

-- Define a function to check if a line is parallel to another line
def isParallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define a function to check if a line intersects triangles in equal segments
def intersectsInEqualSegments (l : Line) (t1 t2 t3 : RightTriangle) : Prop :=
  sorry -- Definition omitted for brevity

-- Main theorem
theorem equal_segments_after_rearrangement
  (l : Line)
  (t1 t2 t3 : RightTriangle)
  (h1 : ∃ (l' : Line), isParallel l l' ∧ intersectsInEqualSegments l' t1 t2 t3) :
  ∃ (l'' : Line), isParallel l l'' ∧ intersectsInEqualSegments l'' t1 t2 t3 :=
by sorry

end NUMINAMATH_CALUDE_equal_segments_after_rearrangement_l455_45564


namespace NUMINAMATH_CALUDE_translate_quadratic_l455_45511

/-- Represents a quadratic function of the form y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies a horizontal and vertical translation to a quadratic function -/
def translate (f : QuadraticFunction) (h : ℝ) (v : ℝ) : QuadraticFunction :=
  { a := f.a
  , b := -2 * f.a * h + f.b
  , c := f.a * h^2 - f.b * h + f.c + v }

theorem translate_quadratic :
  let f : QuadraticFunction := { a := 2, b := 0, c := 0 }
  let g : QuadraticFunction := translate f (-1) 3
  g = { a := 2, b := 4, c := 5 } :=
by sorry

end NUMINAMATH_CALUDE_translate_quadratic_l455_45511


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l455_45539

theorem ratio_x_to_y (x y : ℝ) (h : (14*x - 5*y) / (17*x - 3*y) = 4/6) : x/y = 1/23 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l455_45539


namespace NUMINAMATH_CALUDE_exactly_one_even_l455_45590

theorem exactly_one_even (a b c : ℕ) : 
  ¬((a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨ 
    (a % 2 = 0 ∧ b % 2 = 0) ∨ 
    (a % 2 = 0 ∧ c % 2 = 0) ∨ 
    (b % 2 = 0 ∧ c % 2 = 0)) → 
  ((a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨
   (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨
   (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_even_l455_45590


namespace NUMINAMATH_CALUDE_correct_initial_lives_l455_45514

/-- The number of lives a player starts with in a game -/
def initial_lives : ℕ := 2

/-- The number of extra lives gained in the first level -/
def extra_lives_level1 : ℕ := 6

/-- The number of extra lives gained in the second level -/
def extra_lives_level2 : ℕ := 11

/-- The total number of lives after two levels -/
def total_lives : ℕ := 19

theorem correct_initial_lives :
  initial_lives + extra_lives_level1 + extra_lives_level2 = total_lives :=
by sorry

end NUMINAMATH_CALUDE_correct_initial_lives_l455_45514


namespace NUMINAMATH_CALUDE_solution_to_equation_l455_45503

theorem solution_to_equation (x y : ℝ) : 
  (x - 7)^2 + (y - 8)^2 + (x - y)^2 = 1/3 ↔ x = 7 + 1/3 ∧ y = 7 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l455_45503


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l455_45526

theorem smallest_n_square_and_cube : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (a : ℕ), 5 * n = a^2) ∧ 
  (∃ (b : ℕ), 3 * n = b^3) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), 5 * m = x^2) → 
    (∃ (y : ℕ), 3 * m = y^3) → 
    m ≥ n) ∧
  n = 1125 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l455_45526


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_range_of_a_l455_45557

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a^2 + 1}
def B (a : ℝ) : Set ℝ := {x | x^2 - 3 * (a + 1) * x + 2 * (3 * a + 1) ≤ 0}

-- Define the range of a
def RangeOfA : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3} ∪ {-1}

-- State the theorem
theorem sufficient_condition_implies_range_of_a (a : ℝ) :
  A a ⊆ B a → a ∈ RangeOfA := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_range_of_a_l455_45557


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l455_45588

theorem fraction_zero_implies_x_negative_two (x : ℝ) : 
  (x ≠ 2) → ((|x| - 2) / (x - 2) = 0) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l455_45588


namespace NUMINAMATH_CALUDE_exists_parallelepiped_with_square_coverage_l455_45554

/-- A rectangular parallelepiped with integer dimensions -/
structure Parallelepiped where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- A square with integer side length -/
structure Square where
  side : ℕ+

/-- Represents the coverage of a parallelepiped by three squares -/
structure Coverage where
  parallelepiped : Parallelepiped
  squares : Fin 3 → Square
  covers_without_gaps : Bool
  each_pair_shares_edge : Bool

/-- Theorem stating the existence of a parallelepiped covered by three squares with shared edges -/
theorem exists_parallelepiped_with_square_coverage : 
  ∃ (c : Coverage), c.covers_without_gaps ∧ c.each_pair_shares_edge := by
  sorry

end NUMINAMATH_CALUDE_exists_parallelepiped_with_square_coverage_l455_45554


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l455_45577

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop := x^2 - y^2/m^2 = 1

-- Define the condition that the conjugate axis is twice the transverse axis
def conjugate_twice_transverse (m : ℝ) : Prop := abs m = 2

-- Theorem statement
theorem hyperbola_m_value :
  ∀ m : ℝ, conjugate_twice_transverse m → (m = 2 ∨ m = -2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l455_45577


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l455_45508

/-- Calculate the gain percent for a scooter sale -/
theorem scooter_gain_percent (purchase_price repair_costs selling_price : ℝ) 
  (h1 : purchase_price = 4700)
  (h2 : repair_costs = 800)
  (h3 : selling_price = 6000) : 
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 9.09 :=
by sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_l455_45508


namespace NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l455_45582

theorem pythagorean_triple_divisibility (a b c : ℤ) (h : a^2 + b^2 = c^2) :
  ∃ (k l m : ℤ), 
    (a = 3*k ∨ b = 3*k ∨ c = 3*k) ∧
    (a = 4*l ∨ b = 4*l ∨ c = 4*l) ∧
    (a = 5*m ∨ b = 5*m ∨ c = 5*m) := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l455_45582


namespace NUMINAMATH_CALUDE_mean_of_class_scores_l455_45546

def class_scores : List ℕ := [50, 57, 49, 57, 32, 46, 65, 28]

theorem mean_of_class_scores : 
  (List.sum class_scores) / (List.length class_scores) = 48 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_class_scores_l455_45546


namespace NUMINAMATH_CALUDE_subset_of_countable_is_finite_or_countable_l455_45570

theorem subset_of_countable_is_finite_or_countable 
  (X : Set α) (hX : Countable X) (A : Set α) (hA : A ⊆ X) :
  (Finite A) ∨ (Countable A) :=
sorry

end NUMINAMATH_CALUDE_subset_of_countable_is_finite_or_countable_l455_45570


namespace NUMINAMATH_CALUDE_potato_problem_solution_l455_45591

/-- Represents the potato problem with given conditions --/
def potato_problem (total_potatoes wedge_potatoes wedges_per_potato chips_per_potato : ℕ) : Prop :=
  let remaining_potatoes := total_potatoes - wedge_potatoes
  let chip_potatoes := remaining_potatoes / 2
  let total_chips := chip_potatoes * chips_per_potato
  let total_wedges := wedge_potatoes * wedges_per_potato
  total_chips - total_wedges = 436

/-- Theorem stating the solution to the potato problem --/
theorem potato_problem_solution :
  potato_problem 67 13 8 20 := by
  sorry

#check potato_problem_solution

end NUMINAMATH_CALUDE_potato_problem_solution_l455_45591


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l455_45542

theorem minimum_value_theorem (x y m : ℝ) :
  y ≥ 1 →
  y ≤ 2 * x - 1 →
  x + y ≤ m →
  (∀ x' y' : ℝ, y' ≥ 1 → y' ≤ 2 * x' - 1 → x' + y' ≤ m → x - y ≤ x' - y') →
  x - y = 0 →
  m = 2 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l455_45542


namespace NUMINAMATH_CALUDE_female_lion_weight_l455_45556

theorem female_lion_weight (male_weight : ℚ) (weight_difference : ℚ) :
  male_weight = 145 / 4 →
  weight_difference = 47 / 10 →
  male_weight - weight_difference = 631 / 20 := by
  sorry

end NUMINAMATH_CALUDE_female_lion_weight_l455_45556


namespace NUMINAMATH_CALUDE_slide_problem_l455_45504

theorem slide_problem (initial_boys : ℕ) (total_boys : ℕ) (h1 : initial_boys = 22) (h2 : total_boys = 35) :
  total_boys - initial_boys = 13 := by
  sorry

end NUMINAMATH_CALUDE_slide_problem_l455_45504


namespace NUMINAMATH_CALUDE_overall_profit_percentage_l455_45548

def apples : ℝ := 280
def oranges : ℝ := 150
def bananas : ℝ := 100

def apples_high_profit_ratio : ℝ := 0.4
def oranges_high_profit_ratio : ℝ := 0.45
def bananas_high_profit_ratio : ℝ := 0.5

def apples_high_profit_percentage : ℝ := 0.2
def oranges_high_profit_percentage : ℝ := 0.25
def bananas_high_profit_percentage : ℝ := 0.3

def low_profit_percentage : ℝ := 0.15

def total_fruits : ℝ := apples + oranges + bananas

theorem overall_profit_percentage (ε : ℝ) (h : ε > 0) :
  ∃ (profit_percentage : ℝ),
    abs (profit_percentage - 0.1875) < ε ∧
    profit_percentage = 
      (apples_high_profit_ratio * apples * apples_high_profit_percentage +
       oranges_high_profit_ratio * oranges * oranges_high_profit_percentage +
       bananas_high_profit_ratio * bananas * bananas_high_profit_percentage +
       (1 - apples_high_profit_ratio) * apples * low_profit_percentage +
       (1 - oranges_high_profit_ratio) * oranges * low_profit_percentage +
       (1 - bananas_high_profit_ratio) * bananas * low_profit_percentage) /
      total_fruits :=
by sorry

end NUMINAMATH_CALUDE_overall_profit_percentage_l455_45548


namespace NUMINAMATH_CALUDE_tommys_nickels_l455_45581

/-- The number of coins Tommy has in his collection. -/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  half_dollars : ℕ
  dollar_coins : ℕ

/-- Tommy's coin collection satisfies the given conditions. -/
def valid_collection (c : CoinCollection) : Prop :=
  c.dimes = c.pennies + 10 ∧
  c.nickels = 2 * c.dimes ∧
  c.quarters = 4 ∧
  c.pennies = 10 * c.quarters ∧
  c.half_dollars = c.quarters + 5 ∧
  c.dollar_coins = 3 * c.half_dollars

/-- The number of nickels in Tommy's collection is 100. -/
theorem tommys_nickels (c : CoinCollection) (h : valid_collection c) : c.nickels = 100 := by
  sorry

end NUMINAMATH_CALUDE_tommys_nickels_l455_45581


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l455_45598

/-- Given an ellipse with point P and foci F₁ and F₂, if ∠PF₁F₂ = 60° and |PF₂| = √3|PF₁|,
    then the eccentricity of the ellipse is √3 - 1. -/
theorem ellipse_eccentricity (P F₁ F₂ : ℝ × ℝ) (a c : ℝ) :
  let e := c / a
  let angle_PF₁F₂ := Real.pi / 3  -- 60° in radians
  let dist_PF₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let dist_PF₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  let dist_F₁F₂ := 2 * c
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = 4 * a^2 →  -- P is on the ellipse
  dist_F₁F₂^2 = (F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2 →  -- Definition of distance between foci
  Real.cos angle_PF₁F₂ = (dist_PF₁^2 + dist_F₁F₂^2 - dist_PF₂^2) / (2 * dist_PF₁ * dist_F₁F₂) →  -- Cosine rule
  dist_PF₂ = Real.sqrt 3 * dist_PF₁ →
  e = Real.sqrt 3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l455_45598


namespace NUMINAMATH_CALUDE_circle_has_longest_perimeter_l455_45567

/-- The perimeter of a square with side length 7 cm -/
def square_perimeter : ℝ := 4 * 7

/-- The perimeter of an equilateral triangle with side length 9 cm -/
def triangle_perimeter : ℝ := 3 * 9

/-- The perimeter of a circle with radius 5 cm, using π = 3 -/
def circle_perimeter : ℝ := 2 * 3 * 5

theorem circle_has_longest_perimeter :
  circle_perimeter > square_perimeter ∧ circle_perimeter > triangle_perimeter :=
sorry

end NUMINAMATH_CALUDE_circle_has_longest_perimeter_l455_45567


namespace NUMINAMATH_CALUDE_f_sum_equals_four_l455_45578

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 - b * x^5 + c * x^3 + 2

-- State the theorem
theorem f_sum_equals_four (a b c : ℝ) (h : f a b c (-5) = 3) : f a b c 5 + f a b c (-5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_equals_four_l455_45578


namespace NUMINAMATH_CALUDE_smallest_multiple_with_100_divisors_l455_45519

/-- The number of positive integral divisors of n -/
def divisor_count (n : ℕ) : ℕ := sorry

/-- n is a multiple of m -/
def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

theorem smallest_multiple_with_100_divisors :
  ∃ n : ℕ,
    (∀ m : ℕ, m < n → ¬(is_multiple m 100 ∧ divisor_count m = 100)) ∧
    is_multiple n 100 ∧
    divisor_count n = 100 ∧
    n / 100 = 324 / 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_100_divisors_l455_45519


namespace NUMINAMATH_CALUDE_carter_gave_58_cards_l455_45512

/-- The number of baseball cards Carter gave to Marcus -/
def cards_given (initial final : ℝ) : ℝ := final - initial

/-- Proof that Carter gave Marcus 58 baseball cards -/
theorem carter_gave_58_cards (initial final : ℝ) 
  (h1 : initial = 210.0) 
  (h2 : final = 268) : 
  cards_given initial final = 58 := by
  sorry

end NUMINAMATH_CALUDE_carter_gave_58_cards_l455_45512


namespace NUMINAMATH_CALUDE_union_of_sets_l455_45561

open Set

theorem union_of_sets (M N : Set ℝ) : 
  M = {x : ℝ | 1 < x ∧ x ≤ 3} → 
  N = {x : ℝ | 2 < x ∧ x ≤ 5} → 
  M ∪ N = {x : ℝ | 1 < x ∧ x ≤ 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l455_45561
