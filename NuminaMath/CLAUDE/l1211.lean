import Mathlib

namespace specific_composite_square_perimeter_l1211_121149

/-- Represents a square composed of four rectangles and an inner square -/
structure CompositeSquare where
  /-- Total area of the four rectangles -/
  rectangle_area : ℝ
  /-- Area of the square formed by the inner vertices of the rectangles -/
  inner_square_area : ℝ

/-- Calculates the total perimeter of the four rectangles in a CompositeSquare -/
def total_perimeter (cs : CompositeSquare) : ℝ :=
  sorry

/-- Theorem stating that for a specific CompositeSquare, the total perimeter is 48 -/
theorem specific_composite_square_perimeter :
  ∃ (cs : CompositeSquare),
    cs.rectangle_area = 32 ∧
    cs.inner_square_area = 20 ∧
    total_perimeter cs = 48 :=
  sorry

end specific_composite_square_perimeter_l1211_121149


namespace fourth_root_81_times_cube_root_27_times_sqrt_9_equals_27_l1211_121139

theorem fourth_root_81_times_cube_root_27_times_sqrt_9_equals_27 :
  (81 : ℝ) ^ (1/4) * (27 : ℝ) ^ (1/3) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end fourth_root_81_times_cube_root_27_times_sqrt_9_equals_27_l1211_121139


namespace parabola_focus_l1211_121171

/-- Given a parabola y = ax² passing through (1, 4), its focus is at (0, 1/16) -/
theorem parabola_focus (a : ℝ) : 
  (4 = a * 1^2) → -- Parabola passes through (1, 4)
  let f : ℝ × ℝ := (0, 1/16) -- Define focus coordinates
  (∀ x y : ℝ, y = a * x^2 → -- For all points (x, y) on the parabola
    (x - f.1)^2 = 4 * (1/(4*a)) * (y - f.2)) -- Satisfy the focus-directrix property
  := by sorry

end parabola_focus_l1211_121171


namespace symmetric_points_sum_power_l1211_121161

theorem symmetric_points_sum_power (a b : ℝ) : 
  (∃ (P1 P2 : ℝ × ℝ), 
    P1 = (a - 1, 5) ∧ 
    P2 = (2, b - 1) ∧ 
    P1.1 = P2.1 ∧ 
    P1.2 = -P2.2) →
  (a + b)^2016 = 1 := by
sorry

end symmetric_points_sum_power_l1211_121161


namespace min_value_of_f_l1211_121140

def f (x : ℝ) : ℝ := 
  Finset.sum (Finset.range 2015) (fun i => (i + 1) * x^(2014 - i))

theorem min_value_of_f :
  ∃ (min : ℝ), min = 1008 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

end min_value_of_f_l1211_121140


namespace range_of_m_l1211_121156

-- Define the propositions p and q
def p (x : ℝ) : Prop := -2 ≤ 1 - (x - 1) / 3 ∧ 1 - (x - 1) / 3 ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | ¬(q x m)}
def B : Set ℝ := {x | ¬(p x)}

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  m ≥ 6 :=
sorry

end range_of_m_l1211_121156


namespace cuboid_height_l1211_121147

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the sum of all edges of a cuboid -/
def Cuboid.sumOfEdges (c : Cuboid) : ℝ :=
  4 * (c.length + c.width + c.height)

theorem cuboid_height (c : Cuboid) 
  (h_sum : c.sumOfEdges = 224)
  (h_width : c.width = 30)
  (h_length : c.length = 22) :
  c.height = 4 := by
  sorry

end cuboid_height_l1211_121147


namespace total_chapters_read_l1211_121164

def number_of_books : ℕ := 12
def chapters_per_book : ℕ := 32

theorem total_chapters_read : number_of_books * chapters_per_book = 384 := by
  sorry

end total_chapters_read_l1211_121164


namespace evaluate_expression_l1211_121125

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 1) :
  y^2 * (y - 4*x) = -7 := by
  sorry

end evaluate_expression_l1211_121125


namespace compton_basketball_league_members_l1211_121195

theorem compton_basketball_league_members : 
  let sock_cost : ℚ := 4
  let tshirt_cost : ℚ := sock_cost + 6
  let cap_cost : ℚ := tshirt_cost - 3
  let member_cost : ℚ := 2 * (sock_cost + tshirt_cost + cap_cost)
  let total_expenditure : ℚ := 3144
  (total_expenditure / member_cost : ℚ) = 75 := by
  sorry

end compton_basketball_league_members_l1211_121195


namespace total_panels_eq_600_l1211_121105

/-- The number of houses in the neighborhood -/
def num_houses : ℕ := 10

/-- The number of double windows downstairs in each house -/
def num_double_windows : ℕ := 6

/-- The number of glass panels in each double window -/
def panels_per_double_window : ℕ := 4

/-- The number of single windows upstairs in each house -/
def num_single_windows : ℕ := 8

/-- The number of glass panels in each single window -/
def panels_per_single_window : ℕ := 3

/-- The number of bay windows in each house -/
def num_bay_windows : ℕ := 2

/-- The number of glass panels in each bay window -/
def panels_per_bay_window : ℕ := 6

/-- The total number of glass panels in the neighborhood -/
def total_panels : ℕ := num_houses * (
  num_double_windows * panels_per_double_window +
  num_single_windows * panels_per_single_window +
  num_bay_windows * panels_per_bay_window
)

theorem total_panels_eq_600 : total_panels = 600 := by
  sorry

end total_panels_eq_600_l1211_121105


namespace students_passed_both_tests_l1211_121151

theorem students_passed_both_tests 
  (total : ℕ) 
  (passed_long_jump : ℕ) 
  (passed_shot_put : ℕ) 
  (failed_both : ℕ) 
  (h1 : total = 50)
  (h2 : passed_long_jump = 40)
  (h3 : passed_shot_put = 31)
  (h4 : failed_both = 4) :
  ∃ (passed_both : ℕ), 
    passed_both = 25 ∧ 
    total = passed_both + (passed_long_jump - passed_both) + (passed_shot_put - passed_both) + failed_both :=
by sorry

end students_passed_both_tests_l1211_121151


namespace binary_sum_equality_l1211_121123

/-- Converts a list of bits to a natural number -/
def bitsToNat (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- The sum of the given binary numbers is equal to 11110011₂ -/
theorem binary_sum_equality : 
  let a := bitsToNat [true, false, true, false, true]
  let b := bitsToNat [true, false, true, true]
  let c := bitsToNat [true, true, true, false, false]
  let d := bitsToNat [true, false, true, false, true, false, true]
  let sum := bitsToNat [true, true, true, true, false, false, true, true]
  a + b + c + d = sum := by
  sorry

end binary_sum_equality_l1211_121123


namespace min_value_exponential_quadratic_min_value_achieved_at_zero_l1211_121168

theorem min_value_exponential_quadratic (x : ℝ) : 16^x - 2^x + x^2 + 1 ≥ 1 :=
by
  sorry

theorem min_value_achieved_at_zero : 16^0 - 2^0 + 0^2 + 1 = 1 :=
by
  sorry

end min_value_exponential_quadratic_min_value_achieved_at_zero_l1211_121168


namespace divisible_by_five_unit_digits_l1211_121121

theorem divisible_by_five_unit_digits :
  ∃ (S : Finset Nat), (∀ n : Nat, n % 5 = 0 ↔ n % 10 ∈ S) ∧ Finset.card S = 2 :=
sorry

end divisible_by_five_unit_digits_l1211_121121


namespace lcm_gcf_product_l1211_121154

theorem lcm_gcf_product (a b : ℕ) (ha : a = 36) (hb : b = 48) :
  Nat.lcm a b * Nat.gcd a b = 1728 := by
  sorry

end lcm_gcf_product_l1211_121154


namespace production_time_theorem_l1211_121185

-- Define the time ratios for parts A, B, and C
def time_ratio_A : ℝ := 1
def time_ratio_B : ℝ := 2
def time_ratio_C : ℝ := 3

-- Define the number of parts produced in 10 hours
def parts_A_10h : ℕ := 2
def parts_B_10h : ℕ := 3
def parts_C_10h : ℕ := 4

-- Define the number of parts to be produced
def parts_A_target : ℕ := 14
def parts_B_target : ℕ := 10
def parts_C_target : ℕ := 2

-- Theorem to prove
theorem production_time_theorem :
  ∃ (x : ℝ),
    x > 0 ∧
    x * time_ratio_A * parts_A_10h + x * time_ratio_B * parts_B_10h + x * time_ratio_C * parts_C_10h = 10 ∧
    x * time_ratio_A * parts_A_target + x * time_ratio_B * parts_B_target + x * time_ratio_C * parts_C_target = 20 :=
by
  sorry


end production_time_theorem_l1211_121185


namespace smallest_square_for_five_disks_l1211_121132

/-- A disk with radius 1 -/
structure UnitDisk where
  center : ℝ × ℝ

/-- A square with side length a -/
structure Square (a : ℝ) where
  center : ℝ × ℝ

/-- Predicate to check if two disks overlap -/
def disks_overlap (d1 d2 : UnitDisk) : Prop :=
  (d1.center.1 - d2.center.1)^2 + (d1.center.2 - d2.center.2)^2 < 4

/-- Predicate to check if a disk is contained in a square -/
def disk_in_square (d : UnitDisk) (s : Square a) : Prop :=
  abs (d.center.1 - s.center.1) ≤ a/2 - 1 ∧ abs (d.center.2 - s.center.2) ≤ a/2 - 1

/-- The main theorem -/
theorem smallest_square_for_five_disks :
  ∀ a : ℝ,
  (∃ (s : Square a) (d1 d2 d3 d4 d5 : UnitDisk),
    disk_in_square d1 s ∧ disk_in_square d2 s ∧ disk_in_square d3 s ∧ disk_in_square d4 s ∧ disk_in_square d5 s ∧
    ¬disks_overlap d1 d2 ∧ ¬disks_overlap d1 d3 ∧ ¬disks_overlap d1 d4 ∧ ¬disks_overlap d1 d5 ∧
    ¬disks_overlap d2 d3 ∧ ¬disks_overlap d2 d4 ∧ ¬disks_overlap d2 d5 ∧
    ¬disks_overlap d3 d4 ∧ ¬disks_overlap d3 d5 ∧
    ¬disks_overlap d4 d5) →
  a ≥ 2 + 2 * Real.sqrt 2 :=
by sorry

end smallest_square_for_five_disks_l1211_121132


namespace f_nonnegative_iff_a_eq_four_l1211_121120

/-- The function f(x) = ax³ - 3x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3*x + 1

/-- The set of values for 'a' that satisfy the condition -/
def A : Set ℝ := {a : ℝ | ∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ 0}

theorem f_nonnegative_iff_a_eq_four : A = {4} := by sorry

end f_nonnegative_iff_a_eq_four_l1211_121120


namespace blue_balls_count_l1211_121111

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := sorry

/-- The total number of balls in the bag -/
def total_balls : ℕ := 5 + blue_balls + 4

/-- The probability of picking two red balls -/
def prob_two_red : ℚ := 5 / total_balls * 4 / (total_balls - 1)

theorem blue_balls_count : 
  (5 : ℕ) > 0 ∧ 
  (4 : ℕ) > 0 ∧ 
  prob_two_red = 0.09523809523809523 →
  blue_balls = 6 := by sorry

end blue_balls_count_l1211_121111


namespace letters_written_in_ten_hours_l1211_121138

/-- The number of letters Nathan can write in one hour -/
def nathanRate : ℕ := 25

/-- The number of letters Jacob can write in one hour -/
def jacobRate : ℕ := 2 * nathanRate

/-- The number of hours they write together -/
def totalHours : ℕ := 10

/-- The total number of letters Jacob and Nathan can write together in the given time -/
def totalLetters : ℕ := (nathanRate + jacobRate) * totalHours

theorem letters_written_in_ten_hours : totalLetters = 750 := by
  sorry

end letters_written_in_ten_hours_l1211_121138


namespace height_of_specific_block_l1211_121148

/-- Represents a rectangular block --/
structure RectangularBlock where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The volume of the block in cubic centimeters --/
def volume (block : RectangularBlock) : ℕ :=
  block.length * block.width * block.height

/-- The perimeter of the base of the block in centimeters --/
def basePerimeter (block : RectangularBlock) : ℕ :=
  2 * (block.length + block.width)

theorem height_of_specific_block :
  ∃ (block : RectangularBlock),
    volume block = 42 ∧
    basePerimeter block = 18 ∧
    block.height = 3 :=
by
  sorry

#check height_of_specific_block

end height_of_specific_block_l1211_121148


namespace sum_of_products_l1211_121198

theorem sum_of_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 36) (hac : a * c = 72) (hbc : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 := by
sorry

end sum_of_products_l1211_121198


namespace min_midpoint_for_transformed_sine_l1211_121135

theorem min_midpoint_for_transformed_sine (f g : ℝ → ℝ) (x₁ x₂ : ℝ) :
  (∀ x, f x = Real.sin (x + π/3)) →
  (∀ x, g x = Real.sin (2*x + π/3)) →
  (x₁ ≠ x₂) →
  (g x₁ * g x₂ = -1) →
  (∃ m, m = |(x₁ + x₂)/2| ∧ ∀ y₁ y₂, y₁ ≠ y₂ → g y₁ * g y₂ = -1 → m ≤ |(y₁ + y₂)/2|) →
  |(x₁ + x₂)/2| = π/6 :=
by sorry

end min_midpoint_for_transformed_sine_l1211_121135


namespace opposite_of_negative_2023_l1211_121196

theorem opposite_of_negative_2023 : 
  -((-2023 : ℤ)) = (2023 : ℤ) := by sorry

end opposite_of_negative_2023_l1211_121196


namespace log_equality_l1211_121145

theorem log_equality (c d : ℝ) (hc : c = Real.log 625 / Real.log 4) (hd : d = Real.log 25 / Real.log 5) :
  c = d :=
by sorry

end log_equality_l1211_121145


namespace point_on_coordinate_axes_l1211_121128

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The coordinate axes in a 2D Cartesian coordinate system -/
def CoordinateAxes : Set Point2D :=
  {p : Point2D | p.x = 0 ∨ p.y = 0}

/-- 
Given a point M(a,b) in a Cartesian coordinate system where ab = 0, 
prove that M is located on the coordinate axes.
-/
theorem point_on_coordinate_axes (M : Point2D) (h : M.x * M.y = 0) : 
  M ∈ CoordinateAxes := by
  sorry


end point_on_coordinate_axes_l1211_121128


namespace f_is_quadratic_l1211_121191

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 - x -/
def f (x : ℝ) : ℝ := x^2 - x

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by sorry

end f_is_quadratic_l1211_121191


namespace part_one_part_two_l1211_121166

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem part_one :
  {x : ℝ | f (-1) x ≥ 3} = {x : ℝ | x ≤ -1.5 ∨ x ≥ 1.5} := by sorry

-- Part 2
theorem part_two :
  (∀ x : ℝ, f a x ≥ 2) ↔ (a = 3 ∨ a = -1) := by sorry

end part_one_part_two_l1211_121166


namespace min_angle_in_prime_triangle_l1211_121190

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem min_angle_in_prime_triangle (a b c : ℕ) : 
  (a + b + c = 180) →
  (is_prime a) →
  (is_prime b) →
  (is_prime c) →
  (a > b) →
  (b > c) →
  c ≥ 3 := by
  sorry

end min_angle_in_prime_triangle_l1211_121190


namespace remainder_when_divided_by_fifteen_l1211_121150

theorem remainder_when_divided_by_fifteen (r : ℕ) (h : r / 15 = 82 / 10) : r % 15 = 3 := by
  sorry

end remainder_when_divided_by_fifteen_l1211_121150


namespace orthogonal_vectors_m_value_l1211_121181

/-- Prove that given vectors a = (1,2) and b = (-4,m), if a ⊥ b, then m = 2 -/
theorem orthogonal_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-4, m)
  (a.1 * b.1 + a.2 * b.2 = 0) → m = 2 := by
  sorry

end orthogonal_vectors_m_value_l1211_121181


namespace trigonometric_identity_l1211_121186

theorem trigonometric_identity :
  1 / Real.cos (70 * π / 180) - 2 / Real.sin (70 * π / 180) = 
  4 * Real.sin (10 * π / 180) / Real.sin (40 * π / 180) := by
  sorry

end trigonometric_identity_l1211_121186


namespace k_at_one_eq_neg_155_l1211_121157

/-- Polynomial h(x) -/
def h (p : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + 2*x + 20

/-- Polynomial k(x) -/
def k (q r : ℝ) (x : ℝ) : ℝ := x^4 + 2*x^3 + q*x^2 + 50*x + r

/-- The theorem stating that k(1) = -155 given the conditions -/
theorem k_at_one_eq_neg_155 (p q r : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    h p x = 0 ∧ h p y = 0 ∧ h p z = 0 ∧
    k q r x = 0 ∧ k q r y = 0 ∧ k q r z = 0) →
  k q r 1 = -155 := by
  sorry

end k_at_one_eq_neg_155_l1211_121157


namespace circle_tangent_line_m_values_l1211_121133

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the translation vector
def translation_vector : ℝ × ℝ := (2, 1)

-- Define the translated circle
def translated_circle (x y : ℝ) : Prop :=
  original_circle (x - translation_vector.1) (y - translation_vector.2)

-- Define the tangent line
def tangent_line (x y m : ℝ) : Prop := x + y + m = 0

-- Theorem statement
theorem circle_tangent_line_m_values :
  ∃ m : ℝ, (m = -1 ∨ m = -5) ∧
  ∀ x y : ℝ, translated_circle x y →
  (∃ p : ℝ × ℝ, p.1 + p.2 + m = 0 ∧
  ∀ q : ℝ × ℝ, q.1 + q.2 + m = 0 →
  (p.1 - x)^2 + (p.2 - y)^2 ≤ (q.1 - x)^2 + (q.2 - y)^2) :=
sorry

end circle_tangent_line_m_values_l1211_121133


namespace cat_mouse_problem_l1211_121130

/-- Given that 5 cats can catch 5 mice in 5 minutes, prove that 5 cats can catch 100 mice in 500 minutes -/
theorem cat_mouse_problem (cats mice minutes : ℕ) 
  (h1 : cats = 5)
  (h2 : mice = 5)
  (h3 : minutes = 5)
  (h4 : cats * mice = cats * minutes) : 
  cats * 100 = cats * 500 := by
  sorry

end cat_mouse_problem_l1211_121130


namespace sqrt_sum_problem_l1211_121119

theorem sqrt_sum_problem (x : ℝ) : 
  Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4 → 
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
sorry

end sqrt_sum_problem_l1211_121119


namespace final_amount_after_bets_l1211_121159

theorem final_amount_after_bets (initial_amount : ℝ) (num_bets num_wins num_losses : ℕ) 
  (h1 : initial_amount = 64)
  (h2 : num_bets = 6)
  (h3 : num_wins = 3)
  (h4 : num_losses = 3)
  (h5 : num_wins + num_losses = num_bets) :
  let win_factor := (3/2 : ℝ)
  let loss_factor := (1/2 : ℝ)
  let final_factor := win_factor ^ num_wins * loss_factor ^ num_losses
  initial_amount * final_factor = 27 := by
  sorry

end final_amount_after_bets_l1211_121159


namespace line_point_distance_constraint_l1211_121177

/-- Given a line l: x + y + a = 0 and a point A(2,0), if there exists a point M on line l
    such that |MA| = 2|MO|, then a is in the interval [($2-4\sqrt{2})/3$, ($2+4\sqrt{2})/3$] -/
theorem line_point_distance_constraint (a : ℝ) :
  (∃ x y : ℝ, x + y + a = 0 ∧
    (x - 2)^2 + y^2 = 4 * (x^2 + y^2)) →
  a ∈ Set.Icc ((2 - 4 * Real.sqrt 2) / 3) ((2 + 4 * Real.sqrt 2) / 3) :=
by sorry


end line_point_distance_constraint_l1211_121177


namespace base7_to_base10_65432_l1211_121158

/-- Converts a base-7 number represented as a list of digits to its base-10 equivalent -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base-7 representation of the number -/
def base7Number : List Nat := [2, 3, 4, 5, 6]

/-- Theorem stating that the base-10 equivalent of 65432 in base-7 is 16340 -/
theorem base7_to_base10_65432 :
  base7ToBase10 base7Number = 16340 := by
  sorry

end base7_to_base10_65432_l1211_121158


namespace cody_final_amount_l1211_121122

/-- Given an initial amount, a gift amount, and an expense amount, 
    calculate the final amount of money. -/
def finalAmount (initial gift expense : ℕ) : ℕ :=
  initial + gift - expense

/-- Theorem stating that given the specific values from the problem,
    the final amount is 35 dollars. -/
theorem cody_final_amount : 
  finalAmount 45 9 19 = 35 := by sorry

end cody_final_amount_l1211_121122


namespace sixth_root_unity_product_l1211_121175

theorem sixth_root_unity_product (s : ℂ) (h1 : s^6 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) = 0 := by
  sorry

end sixth_root_unity_product_l1211_121175


namespace alpha_minus_beta_equals_pi_over_four_l1211_121129

open Real

theorem alpha_minus_beta_equals_pi_over_four
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : tan α = 4/3)
  (h4 : tan β = 1/7) :
  α - β = π/4 := by
sorry

end alpha_minus_beta_equals_pi_over_four_l1211_121129


namespace system_solution_l1211_121100

theorem system_solution (m n : ℝ) : 
  (m * 2 + n * 4 = 8 ∧ 2 * m * 2 - 3 * n * 4 = -4) → m = 2 ∧ n = 1 := by
  sorry

end system_solution_l1211_121100


namespace disneyland_arrangements_l1211_121126

theorem disneyland_arrangements (n : ℕ) (k : ℕ) : n = 7 → k = 2 → n.factorial * k^n = 645120 := by
  sorry

end disneyland_arrangements_l1211_121126


namespace emma_age_when_sister_is_56_l1211_121179

theorem emma_age_when_sister_is_56 (emma_current_age : ℕ) (age_difference : ℕ) (sister_future_age : ℕ) :
  emma_current_age = 7 →
  age_difference = 9 →
  sister_future_age = 56 →
  sister_future_age - age_difference = 47 :=
by sorry

end emma_age_when_sister_is_56_l1211_121179


namespace window_width_is_20_inches_l1211_121116

/-- Represents the dimensions of a glass pane -/
structure PaneDimensions where
  width : ℝ
  height : ℝ

/-- Represents the configuration of a window -/
structure WindowConfig where
  pane : PaneDimensions
  columns : ℕ
  rows : ℕ
  borderWidth : ℝ

/-- Calculates the total width of a window given its configuration -/
def totalWidth (config : WindowConfig) : ℝ :=
  config.columns * config.pane.width + (config.columns + 1) * config.borderWidth

/-- Theorem stating the total width of the window is 20 inches -/
theorem window_width_is_20_inches (config : WindowConfig) 
  (h1 : config.columns = 3)
  (h2 : config.rows = 2)
  (h3 : config.pane.height = 3 * config.pane.width)
  (h4 : config.borderWidth = 2) :
  totalWidth config = 20 := by
  sorry

#check window_width_is_20_inches

end window_width_is_20_inches_l1211_121116


namespace volunteer_comprehensive_score_l1211_121152

/-- Calculates the comprehensive score of a volunteer guide based on test scores and weights -/
def comprehensive_score (written_score trial_score interview_score : ℝ)
  (written_weight trial_weight interview_weight : ℝ) : ℝ :=
  written_score * written_weight + trial_score * trial_weight + interview_score * interview_weight

/-- Theorem stating that the comprehensive score of the volunteer guide is 92.4 points -/
theorem volunteer_comprehensive_score :
  comprehensive_score 90 94 92 0.3 0.5 0.2 = 92.4 := by
  sorry

end volunteer_comprehensive_score_l1211_121152


namespace equal_distance_travel_l1211_121118

theorem equal_distance_travel (v1 v2 v3 : ℝ) (t : ℝ) (h1 : v1 = 3) (h2 : v2 = 6) (h3 : v3 = 9) (ht : t = 11/60) :
  let d := t / (1/v1 + 1/v2 + 1/v3)
  3 * d = 0.9 := by sorry

end equal_distance_travel_l1211_121118


namespace arithmetic_sequence_problem_l1211_121141

/-- Arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  arithmetic_sequence a₁ d 11 = -26 →
  arithmetic_sequence a₁ d 51 = 54 →
  (arithmetic_sequence a₁ d 14 = -20) ∧
  (∀ n : ℕ, n < 25 → arithmetic_sequence a₁ d n ≤ 0) ∧
  (arithmetic_sequence a₁ d 25 > 0) := by
  sorry

end arithmetic_sequence_problem_l1211_121141


namespace least_negative_b_for_integer_solutions_l1211_121167

theorem least_negative_b_for_integer_solutions (x b : ℤ) : 
  (∃ x : ℤ, x^2 + b*x = 22) → 
  b < 0 → 
  (∀ b' : ℤ, b' < b → ¬∃ x : ℤ, x^2 + b'*x = 22) →
  b = -21 :=
by sorry

end least_negative_b_for_integer_solutions_l1211_121167


namespace power_function_is_odd_l1211_121184

/-- A function f is a power function if it has the form f(x) = ax^n, where a ≠ 0 and n is a real number. -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x ^ n

/-- A function f is odd if f(-x) = -f(x) for all x in the domain of f. -/
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Given that f(x) = (m - 1)x^(m^2 - 4m + 3) is a power function, prove that f is an odd function. -/
theorem power_function_is_odd (m : ℝ) :
  let f := fun x => (m - 1) * x ^ (m^2 - 4*m + 3)
  isPowerFunction f → isOddFunction f := by
  sorry


end power_function_is_odd_l1211_121184


namespace clock_gains_five_minutes_per_hour_l1211_121104

/-- A clock that gains time -/
structure GainingClock where
  start_time : ℕ  -- Start time in hours (24-hour format)
  end_time : ℕ    -- End time in hours (24-hour format)
  total_gain : ℕ  -- Total minutes gained

/-- Calculate the minutes gained per hour -/
def minutes_gained_per_hour (clock : GainingClock) : ℚ :=
  clock.total_gain / (clock.end_time - clock.start_time)

/-- Theorem: A clock that starts at 9 a.m. and gains 45 minutes by 6 p.m. gains 5 minutes per hour -/
theorem clock_gains_five_minutes_per_hour (clock : GainingClock) 
    (h1 : clock.start_time = 9)
    (h2 : clock.end_time = 18)
    (h3 : clock.total_gain = 45) :
  minutes_gained_per_hour clock = 5 := by
  sorry

end clock_gains_five_minutes_per_hour_l1211_121104


namespace quadratic_roots_shift_l1211_121136

/-- Given a quadratic equation a(x+h)^2+k=0 with roots -3 and 2,
    prove that the roots of a(x-1+h)^2+k=0 are -2 and 3 -/
theorem quadratic_roots_shift (a h k : ℝ) (a_ne_zero : a ≠ 0) :
  (∀ x, a * (x + h)^2 + k = 0 ↔ x = -3 ∨ x = 2) →
  (∀ x, a * (x - 1 + h)^2 + k = 0 ↔ x = -2 ∨ x = 3) :=
by sorry

end quadratic_roots_shift_l1211_121136


namespace decreasing_quadratic_condition_l1211_121174

/-- A function f is decreasing on an interval [a, +∞) if for all x₁, x₂ in [a, +∞) with x₁ < x₂, f(x₁) > f(x₂) -/
def DecreasingOnInterval (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x₁ x₂, a ≤ x₁ → x₁ < x₂ → f x₁ > f x₂

theorem decreasing_quadratic_condition (a : ℝ) :
  DecreasingOnInterval (fun x => a * x^2 + 4 * (a + 1) * x - 3) 2 ↔ a ≤ -1/2 := by
  sorry

end decreasing_quadratic_condition_l1211_121174


namespace geometric_sequence_ratio_two_l1211_121192

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio_two 
  (a : ℕ → ℝ) 
  (h : geometric_sequence a) 
  (h_ratio : ∀ n : ℕ, a (n + 1) = 2 * a n) : 
  (2 * a 1 + a 2) / (2 * a 3 + a 4) = 1 / 4 := by
  sorry

end geometric_sequence_ratio_two_l1211_121192


namespace total_earrings_l1211_121182

theorem total_earrings (bella_earrings : ℕ) (monica_earrings : ℕ) (rachel_earrings : ℕ) 
  (h1 : bella_earrings = 10)
  (h2 : bella_earrings = monica_earrings / 4)
  (h3 : monica_earrings = 2 * rachel_earrings) : 
  bella_earrings + monica_earrings + rachel_earrings = 70 := by
  sorry

end total_earrings_l1211_121182


namespace gcd_factorial_eight_and_factorial_six_squared_l1211_121189

theorem gcd_factorial_eight_and_factorial_six_squared : 
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 11520 := by
  sorry

end gcd_factorial_eight_and_factorial_six_squared_l1211_121189


namespace gcd_of_2750_and_9450_l1211_121153

theorem gcd_of_2750_and_9450 : Nat.gcd 2750 9450 = 50 := by
  sorry

end gcd_of_2750_and_9450_l1211_121153


namespace graph_vertical_shift_l1211_121142

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the vertical translation
def verticalShift (f : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := fun x ↦ f x + c

-- Theorem statement
theorem graph_vertical_shift :
  ∀ (x y : ℝ), y = f x ↔ y + 1 = verticalShift f 1 x := by
  sorry

end graph_vertical_shift_l1211_121142


namespace grain_depot_analysis_l1211_121188

def grain_movements : List Int := [25, -31, -16, 33, -36, -20]
def fee_per_ton : ℕ := 5

theorem grain_depot_analysis :
  (List.sum grain_movements = -45) ∧
  (List.sum (List.map (λ x => fee_per_ton * x.natAbs) grain_movements) = 805) := by
  sorry

end grain_depot_analysis_l1211_121188


namespace line_equation_with_intercept_condition_l1211_121187

/-- The equation of a line passing through the intersection of two given lines,
    with its y-intercept being twice its x-intercept. -/
theorem line_equation_with_intercept_condition :
  ∃ (m b : ℝ),
    (∀ x y : ℝ, (2*x + y = 8 ∧ x - 2*y = -1) → (m*x + b = y)) ∧
    (2 * (b/m) = b) ∧
    ((m = 2 ∧ b = 0) ∨ (m = 2 ∧ b = -8)) := by
  sorry

end line_equation_with_intercept_condition_l1211_121187


namespace income_calculation_l1211_121162

/-- Represents a person's financial situation -/
structure FinancialSituation where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- Proves that given the conditions, the person's income is 10000 -/
theorem income_calculation (f : FinancialSituation) 
  (h1 : f.income * 8 = f.expenditure * 10)  -- income : expenditure = 10 : 8
  (h2 : f.savings = 2000)                   -- savings are 2000
  (h3 : f.income = f.expenditure + f.savings) -- income = expenditure + savings
  : f.income = 10000 := by
  sorry


end income_calculation_l1211_121162


namespace sum_equals_three_halves_l1211_121178

theorem sum_equals_three_halves : 
  let original_sum := (1:ℚ)/3 + 1/5 + 1/7 + 1/9 + 1/11 + 1/13
  let reduced_sum := (1:ℚ)/3 + 1/7 + 1/9 + 1/11
  reduced_sum = 3/2 := by sorry

end sum_equals_three_halves_l1211_121178


namespace right_triangle_perimeter_l1211_121115

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 180 →
  a = 30 →
  c^2 = a^2 + b^2 →
  a + b + c = 42 + 2 * Real.sqrt 261 :=
by sorry

end right_triangle_perimeter_l1211_121115


namespace midpoint_coordinates_sum_l1211_121183

/-- Given that M(-1,6) is the midpoint of CD and C(5,4) is one endpoint, 
    the sum of the coordinates of point D is 1. -/
theorem midpoint_coordinates_sum (C D M : ℝ × ℝ) : 
  C = (5, 4) → 
  M = (-1, 6) → 
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 1 := by
sorry

end midpoint_coordinates_sum_l1211_121183


namespace right_triangle_345_l1211_121103

def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

theorem right_triangle_345 :
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle (Real.sqrt 3) (Real.sqrt 4) (Real.sqrt 5) ∧
  ¬ is_right_triangle 4 6 9 ∧
  is_right_triangle 3 4 5 :=
sorry

end right_triangle_345_l1211_121103


namespace hat_cost_l1211_121197

theorem hat_cost (initial_amount : ℕ) (num_sock_pairs : ℕ) (cost_per_sock_pair : ℕ) (amount_left : ℕ) : 
  initial_amount = 20 ∧ 
  num_sock_pairs = 4 ∧ 
  cost_per_sock_pair = 2 ∧ 
  amount_left = 5 → 
  initial_amount - (num_sock_pairs * cost_per_sock_pair) - amount_left = 7 :=
by sorry

end hat_cost_l1211_121197


namespace annas_cupcake_earnings_l1211_121109

/-- Calculates Anna's earnings from selling cupcakes -/
def annas_earnings (num_trays : ℕ) (cupcakes_per_tray : ℕ) (price_per_cupcake : ℚ) (sold_fraction : ℚ) : ℚ :=
  (num_trays * cupcakes_per_tray : ℚ) * sold_fraction * price_per_cupcake

theorem annas_cupcake_earnings :
  annas_earnings 10 30 (5/2) (7/10) = 525 := by
  sorry

end annas_cupcake_earnings_l1211_121109


namespace unique_rational_root_l1211_121170

def polynomial (x : ℚ) : ℚ := 3 * x^4 - 5 * x^3 - 8 * x^2 + 5 * x + 1

theorem unique_rational_root :
  ∀ x : ℚ, polynomial x = 0 ↔ x = 1/3 := by
  sorry

end unique_rational_root_l1211_121170


namespace solve_complex_equation_l1211_121112

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := 2 - 3 * i * z = -4 + 5 * i * z

-- State the theorem
theorem solve_complex_equation :
  ∃ z : ℂ, equation z ∧ z = -3/4 * i :=
sorry

end solve_complex_equation_l1211_121112


namespace apples_left_l1211_121155

/-- Proves that given the conditions in the problem, the number of boxes of apples left is 3 -/
theorem apples_left (saturday_boxes : Nat) (sunday_boxes : Nat) (apples_per_box : Nat) (apples_sold : Nat) : Nat :=
  by
  -- Define the given conditions
  have h1 : saturday_boxes = 50 := by sorry
  have h2 : sunday_boxes = 25 := by sorry
  have h3 : apples_per_box = 10 := by sorry
  have h4 : apples_sold = 720 := by sorry

  -- Calculate the total number of boxes
  let total_boxes := saturday_boxes + sunday_boxes

  -- Calculate the total number of apples initially
  let total_apples := total_boxes * apples_per_box

  -- Calculate the number of apples left
  let apples_left := total_apples - apples_sold

  -- Calculate the number of boxes left
  let boxes_left := apples_left / apples_per_box

  -- Prove that the number of boxes left is 3
  have h5 : boxes_left = 3 := by sorry

  exact boxes_left

end apples_left_l1211_121155


namespace world_grain_ratio_l1211_121172

def world_grain_supply : ℕ := 1800000
def world_grain_demand : ℕ := 2400000

theorem world_grain_ratio : 
  (world_grain_supply : ℚ) / world_grain_demand = 3 / 4 := by
  sorry

end world_grain_ratio_l1211_121172


namespace max_visible_time_l1211_121137

/-- The maximum time two people can see each other on a circular track with an obstacle -/
theorem max_visible_time (track_radius : ℝ) (obstacle_radius : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : track_radius = 60)
  (h2 : obstacle_radius = 30)
  (h3 : speed1 = 0.4)
  (h4 : speed2 = 0.2) :
  (track_radius * (2 * Real.pi / 3)) / (speed1 - speed2) = 200 * Real.pi := by
  sorry

end max_visible_time_l1211_121137


namespace arithmetic_sequence_constant_ratio_l1211_121102

-- Define the sum of the first n terms of an arithmetic sequence
def T (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 5) / 2

-- State the theorem
theorem arithmetic_sequence_constant_ratio (a : ℚ) :
  (∀ n : ℕ, n > 0 → ∃ k : ℚ, T a (4 * n) / T a n = k) →
  a = 1 / 6 := by
sorry

end arithmetic_sequence_constant_ratio_l1211_121102


namespace infinite_sequence_exists_l1211_121176

-- Define the Ω function
def Omega (n : ℕ+) : ℕ := sorry

-- Define the f function
def f (n : ℕ+) : Int := (-1) ^ (Omega n)

-- State the theorem
theorem infinite_sequence_exists : 
  ∃ (seq : ℕ → ℕ+), (∀ i : ℕ, 
    f (seq i - 1) = 1 ∧ 
    f (seq i) = 1 ∧ 
    f (seq i + 1) = 1) ∧ 
  (∀ i j : ℕ, i ≠ j → seq i ≠ seq j) :=
sorry

end infinite_sequence_exists_l1211_121176


namespace loan_repayment_theorem_l1211_121127

/-- Calculates the lump sum payment for a loan with given parameters -/
def lump_sum_payment (
  principal : ℝ)  -- Initial loan amount
  (rate : ℝ)      -- Annual interest rate as a decimal
  (num_payments : ℕ) -- Total number of annuity payments
  (delay : ℕ)     -- Years before first payment
  (payments_made : ℕ) -- Number of payments made before death
  (years_after_death : ℕ) -- Years after death until lump sum payment
  : ℝ :=
  sorry

theorem loan_repayment_theorem :
  let principal := 20000
  let rate := 0.04
  let num_payments := 10
  let delay := 3
  let payments_made := 5
  let years_after_death := 2
  abs (lump_sum_payment principal rate num_payments delay payments_made years_after_death - 119804.6) < 1 :=
sorry

end loan_repayment_theorem_l1211_121127


namespace factor_x10_minus_1024_l1211_121165

theorem factor_x10_minus_1024 (x : ℝ) : x^10 - 1024 = (x^5 + 32) * (x^5 - 32) := by
  sorry

end factor_x10_minus_1024_l1211_121165


namespace divisibility_by_13_l1211_121194

theorem divisibility_by_13 (x y : ℤ) 
  (h1 : (x^2 - 3*x*y + 2*y^2 + x - y) % 13 = 0)
  (h2 : (x^2 - 2*x*y + y^2 - 5*x + 7) % 13 = 0) :
  (x*y - 12*x + 15*y) % 13 = 0 := by
  sorry

end divisibility_by_13_l1211_121194


namespace work_completion_l1211_121163

theorem work_completion (days1 days2 men2 : ℕ) 
  (h1 : days1 > 0)
  (h2 : days2 > 0)
  (h3 : men2 > 0)
  (h4 : days1 = 80)
  (h5 : days2 = 48)
  (h6 : men2 = 20)
  (h7 : ∃ men1 : ℕ, men1 * days1 = men2 * days2) :
  ∃ men1 : ℕ, men1 = 12 ∧ men1 * days1 = men2 * days2 := by
  sorry

end work_completion_l1211_121163


namespace complex_on_imaginary_axis_l1211_121134

theorem complex_on_imaginary_axis (a : ℝ) : ∃ y : ℝ, (a + I) * (1 + a * I) = y * I := by
  sorry

end complex_on_imaginary_axis_l1211_121134


namespace set_equality_implies_a_equals_three_l1211_121143

theorem set_equality_implies_a_equals_three (a : ℝ) : 
  ({0, 1, a^2} : Set ℝ) = ({1, 0, 2*a + 3} : Set ℝ) → a = 3 := by
  sorry

end set_equality_implies_a_equals_three_l1211_121143


namespace farm_cows_count_l1211_121107

/-- Represents the number of bags of husk eaten by a group of cows in 30 days -/
def total_bags : ℕ := 30

/-- Represents the number of bags of husk eaten by one cow in 30 days -/
def bags_per_cow : ℕ := 1

/-- Calculates the number of cows on the farm -/
def num_cows : ℕ := total_bags / bags_per_cow

/-- Proves that the number of cows on the farm is 30 -/
theorem farm_cows_count : num_cows = 30 := by
  sorry

end farm_cows_count_l1211_121107


namespace sony_games_to_give_away_l1211_121124

theorem sony_games_to_give_away (current_sony_games : ℕ) (target_sony_games : ℕ) :
  current_sony_games = 132 →
  target_sony_games = 31 →
  current_sony_games - target_sony_games = 101 :=
by
  sorry

#check sony_games_to_give_away

end sony_games_to_give_away_l1211_121124


namespace max_value_when_m_neg_four_range_of_m_for_condition_l1211_121108

-- Define the function f
def f (x m : ℝ) : ℝ := x - |x + 2| - |x - 3| - m

-- Theorem for part I
theorem max_value_when_m_neg_four :
  ∃ (x_max : ℝ), ∀ (x : ℝ), f x (-4) ≤ f x_max (-4) ∧ f x_max (-4) = 2 :=
sorry

-- Theorem for part II
theorem range_of_m_for_condition (m : ℝ) :
  (∃ (x₀ : ℝ), f x₀ m ≥ 1 / m - 4) ↔ m ∈ Set.Ioi 0 ∪ {1} :=
sorry

end max_value_when_m_neg_four_range_of_m_for_condition_l1211_121108


namespace pure_imaginary_solutions_l1211_121117

theorem pure_imaginary_solutions (x : ℂ) : 
  (x^5 - 4*x^4 + 6*x^3 - 50*x^2 - 100*x - 120 = 0 ∧ ∃ k : ℝ, x = k*I) ↔ 
  (x = I*Real.sqrt 14 ∨ x = -I*Real.sqrt 14) := by
sorry

end pure_imaginary_solutions_l1211_121117


namespace johns_allowance_problem_l1211_121110

/-- The problem of calculating the fraction of John's remaining allowance spent at the toy store -/
theorem johns_allowance_problem (allowance : ℚ) :
  allowance = 345/100 →
  let arcade_spent := (3/5) * allowance
  let remaining_after_arcade := allowance - arcade_spent
  let candy_spent := 92/100
  let toy_spent := remaining_after_arcade - candy_spent
  (toy_spent / remaining_after_arcade) = 1/3 := by
  sorry

end johns_allowance_problem_l1211_121110


namespace unique_divisor_1058_l1211_121180

theorem unique_divisor_1058 : ∃! d : ℕ, d ≠ 1 ∧ d ∣ 1058 := by sorry

end unique_divisor_1058_l1211_121180


namespace max_grain_mass_on_platform_l1211_121173

/-- Represents a rectangular platform with grain piled on it. -/
structure GrainPlatform where
  length : ℝ
  width : ℝ
  grainDensity : ℝ
  maxAngle : ℝ

/-- Calculates the maximum mass of grain on the platform. -/
def maxGrainMass (platform : GrainPlatform) : ℝ :=
  sorry

/-- Theorem stating the maximum mass of grain on the given platform. -/
theorem max_grain_mass_on_platform :
  let platform : GrainPlatform := {
    length := 8,
    width := 5,
    grainDensity := 1200,
    maxAngle := π/4
  }
  maxGrainMass platform = 47500 := by sorry

end max_grain_mass_on_platform_l1211_121173


namespace min_value_of_S_l1211_121113

theorem min_value_of_S (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 8 := by
  sorry

end min_value_of_S_l1211_121113


namespace problem_solution_l1211_121114

-- Define the sets A, B, and C
def A : Set ℝ := {x | -2 < x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 3 ∨ x < 2}
def C (a : ℝ) : Set ℝ := {x | x < 2 * a + 1}

-- State the theorem
theorem problem_solution :
  (∃ a : ℝ, B ∩ C a = C a) →
  ((A ∩ B = {x : ℝ | -2 < x ∧ x < 2}) ∧
   (∃ a : ℝ, ∀ x : ℝ, x ≤ 1/2 ↔ B ∩ C x = C x)) := by
  sorry

end problem_solution_l1211_121114


namespace opposite_terminal_sides_sin_equality_l1211_121131

theorem opposite_terminal_sides_sin_equality (α β : Real) : 
  (∃ k : Int, β = α + (2 * k + 1) * Real.pi) → |Real.sin α| = |Real.sin β| := by
  sorry

end opposite_terminal_sides_sin_equality_l1211_121131


namespace fraction_simplification_l1211_121144

theorem fraction_simplification : (200 + 10) / (20 + 10) = 7 := by
  sorry

end fraction_simplification_l1211_121144


namespace apple_difference_l1211_121146

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 125

/-- The number of apples Adam has -/
def adam_apples : ℕ := 98

/-- The number of apples Laura has -/
def laura_apples : ℕ := 173

/-- The difference between Laura's apples and the sum of Jackie's and Adam's apples -/
theorem apple_difference : Int.ofNat laura_apples - Int.ofNat (jackie_apples + adam_apples) = -50 := by
  sorry

end apple_difference_l1211_121146


namespace certain_number_equation_l1211_121160

theorem certain_number_equation : ∃ x : ℚ, (55 / 100) * 40 = (4 / 5) * x + 2 :=
by
  -- Proof goes here
  sorry

#check certain_number_equation

end certain_number_equation_l1211_121160


namespace raisin_cost_fraction_l1211_121199

/-- Given a mixture of raisins and nuts with specific quantities and price ratios,
    prove that the cost of raisins is 1/4 of the total cost. -/
theorem raisin_cost_fraction (raisin_pounds almond_pounds cashew_pounds : ℕ) 
                              (raisin_price : ℚ) :
  raisin_pounds = 4 →
  almond_pounds = 3 →
  cashew_pounds = 2 →
  raisin_price > 0 →
  (raisin_pounds * raisin_price) / 
  (raisin_pounds * raisin_price + 
   almond_pounds * (2 * raisin_price) + 
   cashew_pounds * (3 * raisin_price)) = 1 / 4 := by
  sorry

#check raisin_cost_fraction

end raisin_cost_fraction_l1211_121199


namespace combined_price_increase_percentage_l1211_121106

def skateboard_initial_price : ℝ := 120
def knee_pads_initial_price : ℝ := 30
def skateboard_increase_percent : ℝ := 8
def knee_pads_increase_percent : ℝ := 15

theorem combined_price_increase_percentage :
  let skateboard_new_price := skateboard_initial_price * (1 + skateboard_increase_percent / 100)
  let knee_pads_new_price := knee_pads_initial_price * (1 + knee_pads_increase_percent / 100)
  let initial_total := skateboard_initial_price + knee_pads_initial_price
  let new_total := skateboard_new_price + knee_pads_new_price
  (new_total - initial_total) / initial_total * 100 = 9.4 := by sorry

end combined_price_increase_percentage_l1211_121106


namespace ellipse_outside_circle_l1211_121169

theorem ellipse_outside_circle (b : ℝ) (m : ℝ) (x y : ℝ) 
  (h_b : b > 0) (h_m : -1 < m ∧ m < 1) 
  (h_ellipse : x^2 / (b^2 + 1) + y^2 / b^2 = 1) :
  (x - m)^2 + y^2 ≥ 1 - m^2 := by sorry

end ellipse_outside_circle_l1211_121169


namespace total_shark_teeth_l1211_121193

/-- The number of teeth a tiger shark has -/
def tiger_teeth : ℕ := 180

/-- The number of teeth a hammerhead shark has -/
def hammerhead_teeth : ℕ := tiger_teeth / 6

/-- The number of teeth a great white shark has -/
def great_white_teeth : ℕ := 2 * (tiger_teeth + hammerhead_teeth)

/-- The number of teeth a mako shark has -/
def mako_teeth : ℕ := (5 * hammerhead_teeth) / 3

/-- The total number of teeth for all four sharks -/
def total_teeth : ℕ := tiger_teeth + hammerhead_teeth + great_white_teeth + mako_teeth

theorem total_shark_teeth : total_teeth = 680 := by
  sorry

end total_shark_teeth_l1211_121193


namespace triangle_angle_relation_l1211_121101

-- Define the triangle and its properties
structure Triangle where
  A : Real
  B : Real
  C_1 : Real
  C_2 : Real
  B_ext : Real
  h_B_gt_A : B > A
  h_angle_sum : A + B + C_1 + C_2 = 180
  h_ext_angle : B_ext = 180 - B

-- Theorem statement
theorem triangle_angle_relation (t : Triangle) :
  t.C_1 - t.C_2 = t.A + t.B_ext - 180 := by
  sorry

end triangle_angle_relation_l1211_121101
