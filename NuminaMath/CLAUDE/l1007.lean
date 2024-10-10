import Mathlib

namespace function_fits_data_l1007_100707

def f (x : ℝ) : ℝ := 210 - 10*x - x^2 - 2*x^3

theorem function_fits_data : 
  (f 0 = 210) ∧ 
  (f 2 = 170) ∧ 
  (f 4 = 110) ∧ 
  (f 6 = 30) ∧ 
  (f 8 = -70) := by
  sorry

end function_fits_data_l1007_100707


namespace problem_solution_l1007_100702

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

theorem problem_solution (a b : ℝ) :
  (A ∪ B a b = Set.univ) ∧ 
  (A ∩ B a b = Set.Ioc 3 4) →
  a = -3 ∧ b = -4 := by
  sorry

end problem_solution_l1007_100702


namespace cats_count_pet_store_cats_l1007_100798

/-- Given a ratio of cats to dogs and the number of dogs, calculate the number of cats -/
theorem cats_count (cat_ratio : ℕ) (dog_ratio : ℕ) (dog_count : ℕ) : ℕ :=
  (cat_ratio * dog_count) / dog_ratio

/-- Prove that with a cat to dog ratio of 3:4 and 20 dogs, there are 15 cats -/
theorem pet_store_cats : cats_count 3 4 20 = 15 := by
  sorry

end cats_count_pet_store_cats_l1007_100798


namespace gcd_power_minus_one_gcd_fermat_numbers_l1007_100769

-- Part (a)
theorem gcd_power_minus_one (a m n : ℕ) (ha : a > 1) (hm : m ≠ n) :
  Nat.gcd (a^m - 1) (a^n - 1) = a^(Nat.gcd m n) - 1 := by
sorry

-- Part (b)
def fermat (k : ℕ) : ℕ := 2^(2^k) + 1

theorem gcd_fermat_numbers (n m : ℕ) (h : n ≠ m) :
  Nat.gcd (fermat n) (fermat m) = 1 := by
sorry

end gcd_power_minus_one_gcd_fermat_numbers_l1007_100769


namespace frustum_height_theorem_l1007_100781

/-- Represents a pyramid cut by a plane parallel to its base -/
structure CutPyramid where
  -- Height of the original pyramid
  h : ℝ
  -- Height of the smaller pyramid (cut off part)
  h_small : ℝ
  -- Ratio of upper to lower base areas
  area_ratio : ℝ

/-- The height of the frustum in a cut pyramid -/
def frustum_height (p : CutPyramid) : ℝ := p.h - p.h_small

/-- Theorem: If the ratio of upper to lower base areas is 1:4 and the height of the smaller pyramid is 3,
    then the height of the frustum is 3 -/
theorem frustum_height_theorem (p : CutPyramid) 
  (h_ratio : p.area_ratio = 1 / 4)
  (h_small : p.h_small = 3) :
  frustum_height p = 3 := by
  sorry

#check frustum_height_theorem

end frustum_height_theorem_l1007_100781


namespace max_divisor_with_equal_remainders_l1007_100796

theorem max_divisor_with_equal_remainders : 
  ∃ (k : ℕ), 
    (81849 % 243 = k) ∧ 
    (106392 % 243 = k) ∧ 
    (124374 % 243 = k) ∧ 
    (∀ m : ℕ, m > 243 → 
      ¬(∃ r : ℕ, (81849 % m = r) ∧ (106392 % m = r) ∧ (124374 % m = r))) := by
  sorry

end max_divisor_with_equal_remainders_l1007_100796


namespace distance_between_parallel_points_l1007_100704

/-- Given two points A(4, a) and B(5, b) on a line parallel to y = x + m,
    prove that the distance between A and B is √2. -/
theorem distance_between_parallel_points :
  ∀ (a b m : ℝ),
  (b - a) / (5 - 4) = 1 →  -- Parallel condition
  Real.sqrt ((5 - 4)^2 + (b - a)^2) = Real.sqrt 2 := by
sorry

end distance_between_parallel_points_l1007_100704


namespace range_of_two_alpha_l1007_100730

theorem range_of_two_alpha (α β : ℝ) 
  (h1 : π < α + β ∧ α + β < 4 / 3 * π)
  (h2 : -π < α - β ∧ α - β < -π / 3) :
  0 < 2 * α ∧ 2 * α < π :=
by sorry

end range_of_two_alpha_l1007_100730


namespace dice_probability_l1007_100755

def num_dice : ℕ := 8
def num_sides : ℕ := 8
def num_favorable : ℕ := 4

theorem dice_probability :
  let p_first_die : ℚ := (num_favorable : ℚ) / num_sides
  let p_remaining : ℚ := 1 / 2
  let combinations : ℕ := Nat.choose (num_dice - 1) (num_favorable - 1)
  p_first_die * combinations * p_remaining ^ (num_dice - 1) = 35 / 256 := by
    sorry

end dice_probability_l1007_100755


namespace first_number_in_ratio_l1007_100768

theorem first_number_in_ratio (a b : ℕ) : 
  a ≠ 0 → b ≠ 0 → 
  (a : ℚ) / b = 3 / 4 → 
  Nat.lcm a b = 84 → 
  a = 63 :=
by sorry

end first_number_in_ratio_l1007_100768


namespace system_solutions_correct_l1007_100785

theorem system_solutions_correct : 
  -- System 1
  (∃ (x y : ℝ), x - 2*y = 1 ∧ 3*x + 2*y = 7 ∧ x = 2 ∧ y = 1/2) ∧
  -- System 2
  (∃ (x y : ℝ), x - y = 3 ∧ (x - y - 3)/2 - y/3 = -1 ∧ x = 6 ∧ y = 3) :=
by sorry

end system_solutions_correct_l1007_100785


namespace sqrt_equation_solutions_l1007_100788

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt (3 - x) + Real.sqrt (x - 2) = 2) ↔ (x = 3/4 ∨ x = 2) :=
by sorry

end sqrt_equation_solutions_l1007_100788


namespace gcd_a_squared_plus_9a_plus_24_and_a_plus_4_l1007_100729

theorem gcd_a_squared_plus_9a_plus_24_and_a_plus_4 (a : ℤ) (h : ∃ k : ℤ, a = 1428 * k) :
  Nat.gcd (Int.natAbs (a^2 + 9*a + 24)) (Int.natAbs (a + 4)) = 4 := by
  sorry

end gcd_a_squared_plus_9a_plus_24_and_a_plus_4_l1007_100729


namespace lunch_cakes_count_l1007_100725

def total_cakes : ℕ := 15
def dinner_cakes : ℕ := 9

theorem lunch_cakes_count : total_cakes - dinner_cakes = 6 := by
  sorry

end lunch_cakes_count_l1007_100725


namespace robert_reading_capacity_l1007_100778

/-- The number of books Robert can read given his reading speed, book length, and available time -/
def books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (available_hours : ℕ) : ℕ :=
  (pages_per_hour * available_hours) / pages_per_book

/-- Theorem stating that Robert can read 2 books in 8 hours -/
theorem robert_reading_capacity :
  books_read 100 400 8 = 2 := by
  sorry

#eval books_read 100 400 8

end robert_reading_capacity_l1007_100778


namespace chocolate_bars_in_large_box_l1007_100782

theorem chocolate_bars_in_large_box : 
  ∀ (num_small_boxes : ℕ) (bars_per_small_box : ℕ),
    num_small_boxes = 15 →
    bars_per_small_box = 20 →
    num_small_boxes * bars_per_small_box = 300 :=
by
  sorry

end chocolate_bars_in_large_box_l1007_100782


namespace max_dot_product_on_ellipses_l1007_100753

def ellipse_C1 (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

def ellipse_C2 (x y : ℝ) : Prop := x^2 / 9 + y^2 / 25 = 1

def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem max_dot_product_on_ellipses :
  ∀ x1 y1 x2 y2 : ℝ,
  ellipse_C1 x1 y1 → ellipse_C2 x2 y2 →
  dot_product x1 y1 x2 y2 ≤ 15 :=
by sorry

end max_dot_product_on_ellipses_l1007_100753


namespace correct_calculation_l1007_100766

theorem correct_calculation (x : ℝ) (h : 15 * x = 45) : 5 * x = 15 := by
  sorry

end correct_calculation_l1007_100766


namespace solution_sets_imply_a_minus_b_eq_neg_seven_l1007_100724

-- Define the solution sets A and B
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | x^2 - 5*x + 4 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the coefficients a and b
def a : ℝ := -4
def b : ℝ := 3

-- Theorem statement
theorem solution_sets_imply_a_minus_b_eq_neg_seven :
  (A_intersect_B = {x | x^2 + a*x + b < 0}) →
  a - b = -7 := by sorry

end solution_sets_imply_a_minus_b_eq_neg_seven_l1007_100724


namespace bucket_capacity_problem_l1007_100720

/-- Proves that if reducing a bucket's capacity to 4/5 of its original requires 250 buckets to fill a tank, 
    then the number of buckets needed with the original capacity is 200. -/
theorem bucket_capacity_problem (tank_volume : ℝ) (original_capacity : ℝ) 
  (h1 : tank_volume > 0) (h2 : original_capacity > 0) :
  (tank_volume = 250 * (4/5 * original_capacity)) → 
  (tank_volume = 200 * original_capacity) :=
by
  sorry

#check bucket_capacity_problem

end bucket_capacity_problem_l1007_100720


namespace only_negative_three_less_than_reciprocal_l1007_100752

def is_less_than_reciprocal (x : ℝ) : Prop :=
  x ≠ 0 ∧ x < 1 / x

theorem only_negative_three_less_than_reciprocal :
  (is_less_than_reciprocal (-3)) ∧
  (¬ is_less_than_reciprocal (-1/2)) ∧
  (¬ is_less_than_reciprocal 0) ∧
  (¬ is_less_than_reciprocal 1) ∧
  (¬ is_less_than_reciprocal (3/2)) :=
by sorry

end only_negative_three_less_than_reciprocal_l1007_100752


namespace number_multiplied_by_five_thirds_l1007_100759

theorem number_multiplied_by_five_thirds : ∃ x : ℚ, (5 : ℚ) / 3 * x = 45 ∧ x = 27 := by
  sorry

end number_multiplied_by_five_thirds_l1007_100759


namespace calculation_one_l1007_100717

theorem calculation_one : (1) - 2 + (-3) - (-5) + 7 = 7 := by
  sorry

end calculation_one_l1007_100717


namespace cards_selection_count_l1007_100745

/-- The number of ways to select 3 cards from 12 cards (3 each of red, yellow, green, and blue) 
    such that they are not all the same color and there is at most 1 blue card. -/
def select_cards : ℕ := sorry

/-- The total number of cards -/
def total_cards : ℕ := 12

/-- The number of cards of each color -/
def cards_per_color : ℕ := 3

/-- The number of colors -/
def num_colors : ℕ := 4

/-- The number of cards to be selected -/
def cards_to_select : ℕ := 3

theorem cards_selection_count : 
  select_cards = Nat.choose total_cards cards_to_select - 
                 num_colors - 
                 (Nat.choose cards_per_color 2 * Nat.choose (total_cards - cards_per_color) 1) := by
  sorry

end cards_selection_count_l1007_100745


namespace managing_team_selection_l1007_100744

def society_size : ℕ := 20
def team_size : ℕ := 3

theorem managing_team_selection :
  Nat.choose society_size team_size = 1140 := by
  sorry

end managing_team_selection_l1007_100744


namespace problem_solution_l1007_100731

open Real

/-- The function f(x) = e^x + sin(x) + b -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := exp x + sin x + b

/-- The function g(x) = xe^x -/
noncomputable def g (x : ℝ) : ℝ := x * exp x

theorem problem_solution :
  (∀ b : ℝ, (∀ x : ℝ, x ≥ 0 → f b x ≥ 0) → b ≥ -1) ∧
  (∀ m : ℝ, (∃ b : ℝ, (∀ x : ℝ, exp x + b = x - 1) ∧
                     (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ exp x₁ - 2 = (m - 2*x₁)/x₁ ∧
                                   exp x₂ - 2 = (m - 2*x₂)/x₂) ∧
                     (∀ x : ℝ, exp x - 2 = (m - 2*x)/x → x = x₁ ∨ x = x₂)) →
   -1/exp 1 < m ∧ m < 0) :=
by sorry

end problem_solution_l1007_100731


namespace range_of_f_range_of_a_l1007_100771

-- Define the function f
def f (x : ℝ) : ℝ := 2 * |x - 1| - |x - 4|

-- Theorem for the range of f
theorem range_of_f : Set.range f = Set.Ici (-3) := by sorry

-- Define the inequality function g
def g (x a : ℝ) : ℝ := 2 * |x - 1| - |x - a|

-- Theorem for the range of a
theorem range_of_a : ∀ a : ℝ, (∀ x : ℝ, g x a ≥ -1) ↔ a ∈ Set.Icc 0 2 := by sorry

end range_of_f_range_of_a_l1007_100771


namespace matrix_power_2023_l1007_100793

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end matrix_power_2023_l1007_100793


namespace composite_numbers_1991_l1007_100706

theorem composite_numbers_1991 : 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = 1991^1991 + 1) ∧ 
  (∃ c d : ℕ, c > 1 ∧ d > 1 ∧ c * d = 1991^1991 - 1) := by
  sorry

end composite_numbers_1991_l1007_100706


namespace correct_operation_l1007_100757

theorem correct_operation (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end correct_operation_l1007_100757


namespace cubeTowerSurfaceAreaIs1221_l1007_100715

/-- Calculates the surface area of a cube tower given a list of cube side lengths -/
def cubeTowerSurfaceArea (sideLengths : List ℕ) : ℕ :=
  match sideLengths with
  | [] => 0
  | [x] => 6 * x^2
  | x :: xs => 4 * x^2 + cubeTowerSurfaceArea xs

/-- The list of cube side lengths in the tower -/
def towerSideLengths : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- Theorem stating that the surface area of the cube tower is 1221 square units -/
theorem cubeTowerSurfaceAreaIs1221 :
  cubeTowerSurfaceArea towerSideLengths = 1221 := by
  sorry


end cubeTowerSurfaceAreaIs1221_l1007_100715


namespace first_quadrant_iff_sin_cos_sum_gt_one_l1007_100735

/-- An angle is in the first quadrant if it's between 0 and π/2 radians -/
def is_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

/-- The main theorem stating the equivalence between an angle being in the first quadrant
    and the sum of its sine and cosine being greater than 1 -/
theorem first_quadrant_iff_sin_cos_sum_gt_one (α : ℝ) :
  is_first_quadrant α ↔ Real.sin α + Real.cos α > 1 :=
sorry

end first_quadrant_iff_sin_cos_sum_gt_one_l1007_100735


namespace cricket_run_rate_proof_l1007_100774

/-- Calculates the required run rate for the remaining overs in a cricket game. -/
def required_run_rate (total_overs : ℕ) (first_overs : ℕ) (first_run_rate : ℚ) (target : ℕ) : ℚ :=
  let remaining_overs := total_overs - first_overs
  let first_runs := (first_run_rate * first_overs : ℚ).floor
  let remaining_runs := target - first_runs
  (remaining_runs : ℚ) / remaining_overs

/-- Proves that the required run rate for the remaining 40 overs is 6.5 -/
theorem cricket_run_rate_proof :
  required_run_rate 50 10 (32/10) 292 = 13/2 := by
  sorry

end cricket_run_rate_proof_l1007_100774


namespace max_chords_for_ten_points_l1007_100748

/-- Given n points on a circle, max_chords_no_triangle calculates the maximum number of chords
    that can be drawn between these points without forming any triangles. -/
def max_chords_no_triangle (n : ℕ) : ℕ :=
  (n^2) / 4

/-- Theorem stating that for 10 points on a circle, the maximum number of chords
    that can be drawn without forming triangles is 25. -/
theorem max_chords_for_ten_points :
  max_chords_no_triangle 10 = 25 :=
by sorry

end max_chords_for_ten_points_l1007_100748


namespace x_gt_2_necessary_not_sufficient_for_x_gt_5_l1007_100797

theorem x_gt_2_necessary_not_sufficient_for_x_gt_5 :
  (∀ x : ℝ, x > 5 → x > 2) ∧ (∃ x : ℝ, x > 2 ∧ x ≤ 5) := by
  sorry

end x_gt_2_necessary_not_sufficient_for_x_gt_5_l1007_100797


namespace exist_point_W_l1007_100719

-- Define the triangle XYZ
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := X
  let (x₂, y₂) := Y
  let (x₃, y₃) := Z
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 10^2 ∧
  (x₂ - x₃)^2 + (y₂ - y₃)^2 = 11^2 ∧
  (x₁ - x₃)^2 + (y₁ - y₃)^2 = 12^2

-- Define point P on XZ
def PointP (X Z P : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := X
  let (x₃, y₃) := Z
  let (xp, yp) := P
  (xp - x₃)^2 + (yp - y₃)^2 = 6^2 ∧
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ xp = t * x₁ + (1 - t) * x₃ ∧ yp = t * y₁ + (1 - t) * y₃

-- Define point W on line PY
def PointW (Y P W : ℝ × ℝ) : Prop :=
  let (x₂, y₂) := Y
  let (xp, yp) := P
  let (xw, yw) := W
  ∃ t : ℝ, xw = t * xp + (1 - t) * x₂ ∧ yw = t * yp + (1 - t) * y₂

-- Define XW parallel to ZY
def Parallel (X W Z Y : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := X
  let (xw, yw) := W
  let (x₃, y₃) := Z
  let (x₂, y₂) := Y
  (xw - x₁) * (y₃ - y₂) = (yw - y₁) * (x₃ - x₂)

-- Define cyclic hexagon
def CyclicHexagon (Y X Y Z W X : ℝ × ℝ) : Prop :=
  -- This is a simplified definition, as the full condition for a cyclic hexagon is complex
  -- In reality, we would need to check if all six points lie on a circle
  true

-- Main theorem
theorem exist_point_W (X Y Z P : ℝ × ℝ) :
  Triangle X Y Z →
  PointP X Z P →
  ∃ W : ℝ × ℝ,
    PointW Y P W ∧
    Parallel X W Z Y ∧
    CyclicHexagon Y X Y Z W X ∧
    let (xp, yp) := P
    let (xw, yw) := W
    (xw - xp)^2 + (yw - yp)^2 = 10^2 :=
by sorry

end exist_point_W_l1007_100719


namespace janet_final_lives_l1007_100736

/-- Calculates the final number of lives for Janet in the video game --/
def final_lives (initial_lives : ℕ) (lives_lost : ℕ) (points_earned : ℕ) : ℕ :=
  let remaining_lives := initial_lives - lives_lost
  let lives_earned := (points_earned / 100) * 2
  let lives_lost_penalty := points_earned / 200
  remaining_lives + lives_earned - lives_lost_penalty

theorem janet_final_lives : 
  final_lives 47 23 1840 = 51 := by
  sorry

end janet_final_lives_l1007_100736


namespace soda_distribution_impossibility_l1007_100767

theorem soda_distribution_impossibility (total_sodas : ℕ) (sisters : ℕ) : 
  total_sodas = 9 →
  sisters = 2 →
  ¬∃ (sodas_per_sibling : ℕ), 
    sodas_per_sibling > 0 ∧ 
    total_sodas = sodas_per_sibling * (sisters + 2 * sisters) :=
by
  sorry

end soda_distribution_impossibility_l1007_100767


namespace l_shaped_area_l1007_100710

/-- The area of an L-shaped region formed by subtracting two smaller squares from a larger square --/
theorem l_shaped_area (square_side : ℝ) (small_square1_side : ℝ) (small_square2_side : ℝ)
  (h1 : square_side = 6)
  (h2 : small_square1_side = 2)
  (h3 : small_square2_side = 3)
  (h4 : small_square1_side < square_side)
  (h5 : small_square2_side < square_side) :
  square_side^2 - small_square1_side^2 - small_square2_side^2 = 23 := by
  sorry

#check l_shaped_area

end l_shaped_area_l1007_100710


namespace valid_paintings_count_l1007_100750

/-- Represents a color in the painting. -/
inductive Color
  | Green
  | Red
  | Blue

/-- Represents a position in the 3x3 grid. -/
structure Position :=
  (row : Fin 3)
  (col : Fin 3)

/-- Represents a painting of the 3x3 grid. -/
def Painting := Position → Color

/-- Checks if a painting satisfies the color placement rules. -/
def validPainting (p : Painting) : Prop :=
  ∀ (pos : Position),
    (p pos = Color.Green →
      ∀ (above : Position), above.row = pos.row - 1 → above.col = pos.col → p above ≠ Color.Red) ∧
    (p pos = Color.Green →
      ∀ (right : Position), right.row = pos.row → right.col = pos.col + 1 → p right ≠ Color.Red) ∧
    (p pos = Color.Blue →
      ∀ (left : Position), left.row = pos.row → left.col = pos.col - 1 → p left ≠ Color.Red)

/-- The number of valid paintings. -/
def numValidPaintings : ℕ := sorry

theorem valid_paintings_count :
  numValidPaintings = 78 :=
sorry

end valid_paintings_count_l1007_100750


namespace triangle_angle_sixty_degrees_l1007_100786

theorem triangle_angle_sixty_degrees (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C →
  C = Real.pi / 3 := by
sorry

end triangle_angle_sixty_degrees_l1007_100786


namespace ball_distribution_theorem_l1007_100758

/-- The number of ways to distribute n different balls into k different boxes -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n different balls into k different boxes with exactly m empty boxes -/
def distributeWithEmpty (n k m : ℕ) : ℕ := sorry

theorem ball_distribution_theorem (n k : ℕ) (hn : n = 4) (hk : k = 4) :
  distribute n k = 256 ∧
  distributeWithEmpty n k 1 = 144 ∧
  distributeWithEmpty n k 2 = 84 := by sorry

end ball_distribution_theorem_l1007_100758


namespace sector_arc_length_l1007_100779

theorem sector_arc_length (r : ℝ) (θ_deg : ℝ) (L : ℝ) : 
  r = 1 → θ_deg = 60 → L = r * (θ_deg * π / 180) → L = π / 3 := by
  sorry

end sector_arc_length_l1007_100779


namespace sum_of_derivatives_positive_l1007_100780

def f (x : ℝ) : ℝ := -x^2 - x^4 - x^6

theorem sum_of_derivatives_positive (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ < 0) (h₂ : x₂ + x₃ < 0) (h₃ : x₃ + x₁ < 0) : 
  (deriv f x₁) + (deriv f x₂) + (deriv f x₃) > 0 := by
  sorry

end sum_of_derivatives_positive_l1007_100780


namespace keith_books_l1007_100703

theorem keith_books (jason_books : ℕ) (total_books : ℕ) (h1 : jason_books = 21) (h2 : total_books = 41) :
  total_books - jason_books = 20 := by
  sorry

end keith_books_l1007_100703


namespace min_sum_same_last_three_digits_l1007_100776

/-- Given two positive integers m and n where n > m ≥ 1, this theorem states that
    if 1978^n and 1978^m have the same last three digits, then m + n ≥ 106. -/
theorem min_sum_same_last_three_digits (m n : ℕ) (hm : m ≥ 1) (hn : n > m) :
  (1978^n : ℕ) % 1000 = (1978^m : ℕ) % 1000 → m + n ≥ 106 := by
  sorry

end min_sum_same_last_three_digits_l1007_100776


namespace danny_fish_tank_theorem_l1007_100732

/-- The number of fish remaining after selling some from Danny's fish tank. -/
def remaining_fish (initial_guppies initial_angelfish initial_tiger_sharks initial_oscar_fish
                    sold_guppies sold_angelfish sold_tiger_sharks sold_oscar_fish : ℕ) : ℕ :=
  (initial_guppies - sold_guppies) +
  (initial_angelfish - sold_angelfish) +
  (initial_tiger_sharks - sold_tiger_sharks) +
  (initial_oscar_fish - sold_oscar_fish)

/-- Theorem stating the number of remaining fish in Danny's tank. -/
theorem danny_fish_tank_theorem :
  remaining_fish 94 76 89 58 30 48 17 24 = 198 := by
  sorry

end danny_fish_tank_theorem_l1007_100732


namespace trigonometric_inequality_l1007_100733

theorem trigonometric_inequality (x : ℝ) (n m : ℕ) 
  (h1 : 0 < x) (h2 : x < Real.pi / 2) (h3 : n > m) :
  2 * |Real.sin x ^ n - Real.cos x ^ n| ≤ 3 * |Real.sin x ^ m - Real.cos x ^ m| := by
sorry

end trigonometric_inequality_l1007_100733


namespace quadratic_coefficient_l1007_100772

/-- A quadratic function with vertex (3, 2) passing through (-2, -18) has a = -4/5 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Condition 1
  (2 = a * 3^2 + b * 3 + c) →             -- Condition 2 (vertex)
  (3 = -b / (2 * a)) →                    -- Condition 2 (vertex x-coordinate)
  (-18 = a * (-2)^2 + b * (-2) + c) →     -- Condition 3
  a = -4/5 := by sorry

end quadratic_coefficient_l1007_100772


namespace orange_boxes_total_l1007_100794

theorem orange_boxes_total (box1_capacity box2_capacity box3_capacity : ℕ)
  (box1_fill box2_fill box3_fill : ℚ) :
  box1_capacity = 80 →
  box2_capacity = 50 →
  box3_capacity = 60 →
  box1_fill = 3/4 →
  box2_fill = 3/5 →
  box3_fill = 2/3 →
  (↑box1_capacity * box1_fill + ↑box2_capacity * box2_fill + ↑box3_capacity * box3_fill : ℚ) = 130 := by
  sorry

end orange_boxes_total_l1007_100794


namespace gcd_of_product_of_differences_l1007_100764

theorem gcd_of_product_of_differences (a b c d : ℤ) : 
  ∃ (k : ℤ), (12 : ℤ) ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) ∧
  ∀ (m : ℤ), (∀ (x y z w : ℤ), m ∣ (x - y) * (x - z) * (x - w) * (y - z) * (y - w) * (z - w)) → m ∣ 12 := by
  sorry

end gcd_of_product_of_differences_l1007_100764


namespace circle_area_with_diameter_9_l1007_100751

theorem circle_area_with_diameter_9 (π : Real) (h : π = Real.pi) :
  let d := 9
  let r := d / 2
  let area := π * r^2
  area = π * (9/2)^2 := by sorry

end circle_area_with_diameter_9_l1007_100751


namespace medication_dosage_range_l1007_100763

theorem medication_dosage_range 
  (daily_min : ℝ) 
  (daily_max : ℝ) 
  (num_doses : ℕ) 
  (h1 : daily_min = 60) 
  (h2 : daily_max = 120) 
  (h3 : num_doses = 4) :
  ∃ x_min x_max : ℝ, 
    x_min = daily_min / num_doses ∧ 
    x_max = daily_max / num_doses ∧ 
    x_min = 15 ∧ 
    x_max = 30 ∧ 
    ∀ x : ℝ, (x_min ≤ x ∧ x ≤ x_max) ↔ (15 ≤ x ∧ x ≤ 30) :=
by sorry

end medication_dosage_range_l1007_100763


namespace line_equation_theorem_l1007_100723

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the area of the triangle formed by a line and the coordinate axes -/
def triangleArea (l : Line) : ℝ := sorry

/-- Check if a line passes through a given point -/
def passesThrough (l : Line) (x y : ℝ) : Prop := 
  l.a * x + l.b * y + l.c = 0

/-- The main theorem -/
theorem line_equation_theorem (l : Line) :
  triangleArea l = 3 ∧ passesThrough l (-3) 4 →
  (l.a = 2 ∧ l.b = 3 ∧ l.c = -6) ∨ (l.a = 8 ∧ l.b = 3 ∧ l.c = 12) := by
  sorry

end line_equation_theorem_l1007_100723


namespace goods_train_speed_l1007_100789

theorem goods_train_speed 
  (man_train_speed : ℝ) 
  (goods_train_length : ℝ) 
  (passing_time : ℝ) 
  (h1 : man_train_speed = 70) 
  (h2 : goods_train_length = 0.28) 
  (h3 : passing_time = 9 / 3600) : 
  ∃ (goods_train_speed : ℝ), 
    goods_train_speed = 42 ∧ 
    (goods_train_speed + man_train_speed) * passing_time = goods_train_length :=
by sorry

end goods_train_speed_l1007_100789


namespace project_work_time_difference_l1007_100718

/-- Given three people working on a project with their working times in the ratio of 3:5:6,
    and a total project time of 140 hours, prove that the difference between the working hours
    of the person who worked the most and the person who worked the least is 30 hours. -/
theorem project_work_time_difference :
  ∀ (x : ℝ), 
  (3 * x + 5 * x + 6 * x = 140) →
  (6 * x - 3 * x = 30) :=
by sorry

end project_work_time_difference_l1007_100718


namespace library_average_disk_space_per_hour_l1007_100790

/-- Represents a digital music library -/
structure MusicLibrary where
  days : ℕ
  diskSpace : ℕ

/-- Calculates the average disk space usage per hour for a given music library -/
def averageDiskSpacePerHour (library : MusicLibrary) : ℚ :=
  library.diskSpace / (library.days * 24)

/-- Theorem stating that for the given library, the average disk space per hour is 50 MB -/
theorem library_average_disk_space_per_hour :
  let library : MusicLibrary := { days := 15, diskSpace := 18000 }
  averageDiskSpacePerHour library = 50 := by
  sorry

end library_average_disk_space_per_hour_l1007_100790


namespace marcus_pretzels_l1007_100711

theorem marcus_pretzels (total : ℕ) (john : ℕ) (alan : ℕ) (marcus : ℕ) 
  (h1 : total = 95)
  (h2 : john = 28)
  (h3 : alan = john - 9)
  (h4 : marcus = john + 12) :
  marcus = 40 := by
  sorry

end marcus_pretzels_l1007_100711


namespace uniform_price_calculation_l1007_100787

/-- Represents the price of the uniform in Rupees -/
def uniform_price : ℕ := 25

/-- Represents the full year salary in Rupees -/
def full_year_salary : ℕ := 900

/-- Represents the number of months served -/
def months_served : ℕ := 9

/-- Represents the actual payment received for the partial service in Rupees -/
def partial_payment : ℕ := 650

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

theorem uniform_price_calculation :
  uniform_price = (full_year_salary * months_served / months_in_year) - partial_payment := by
  sorry

end uniform_price_calculation_l1007_100787


namespace sqrt_equation_solution_l1007_100705

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (4 * x + 15 - 6) = 12 → x = 33.75 := by
  sorry

end sqrt_equation_solution_l1007_100705


namespace fraction_inequality_l1007_100749

theorem fraction_inequality (a b c d e : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c < d)
  (h4 : d < 0)
  (h5 : e < 0) :
  e / (a - c) > e / (b - d) := by
  sorry

end fraction_inequality_l1007_100749


namespace sqrt_8_minus_abs_neg_2_plus_reciprocal_1_3_minus_2_cos_45_l1007_100747

theorem sqrt_8_minus_abs_neg_2_plus_reciprocal_1_3_minus_2_cos_45 :
  Real.sqrt 8 - abs (-2) + (1/3)⁻¹ - 2 * Real.cos (45 * π / 180) = Real.sqrt 2 + 1 := by
  sorry

end sqrt_8_minus_abs_neg_2_plus_reciprocal_1_3_minus_2_cos_45_l1007_100747


namespace solve_letter_problem_l1007_100775

def letter_problem (brother_letters : ℕ) (greta_extra : ℕ) : Prop :=
  let greta_letters := brother_letters + greta_extra
  let total_greta_brother := brother_letters + greta_letters
  let mother_letters := 2 * total_greta_brother
  let total_letters := brother_letters + greta_letters + mother_letters
  (brother_letters = 40) ∧ (greta_extra = 10) → (total_letters = 270)

theorem solve_letter_problem : letter_problem 40 10 := by
  sorry

end solve_letter_problem_l1007_100775


namespace sum_of_squares_l1007_100773

theorem sum_of_squares (x y z a b c : ℝ) 
  (h1 : x * y = a) 
  (h2 : x * z = b) 
  (h3 : y * z = c) 
  (h4 : x ≠ 0) 
  (h5 : y ≠ 0) 
  (h6 : z ≠ 0) 
  (h7 : a ≠ 0) 
  (h8 : b ≠ 0) 
  (h9 : c ≠ 0) : 
  x^2 + y^2 + z^2 = ((a*b)^2 + (a*c)^2 + (b*c)^2) / (a*b*c) := by
sorry

end sum_of_squares_l1007_100773


namespace road_trip_distance_l1007_100765

/-- Road trip problem -/
theorem road_trip_distance (total_distance michelle_distance : ℕ) 
  (h1 : total_distance = 1000)
  (h2 : michelle_distance = 294)
  (h3 : ∃ (tracy_distance : ℕ), tracy_distance > 2 * michelle_distance)
  (h4 : ∃ (katie_distance : ℕ), michelle_distance = 3 * katie_distance) :
  ∃ (tracy_distance : ℕ), tracy_distance = total_distance - michelle_distance - (michelle_distance / 3) ∧ 
    tracy_distance - 2 * michelle_distance = 20 := by
  sorry

end road_trip_distance_l1007_100765


namespace square_plus_inverse_square_l1007_100740

theorem square_plus_inverse_square (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 := by
  sorry

end square_plus_inverse_square_l1007_100740


namespace min_value_a_l1007_100738

theorem min_value_a (a b : ℤ) (m : ℕ) (h1 : a - b = m) (h2 : Nat.Prime m) 
  (h3 : ∃ n : ℕ, a * b = n * n) (h4 : a ≥ 2012) : 
  (∀ a' b' : ℤ, ∃ m' : ℕ, a' - b' = m' ∧ Nat.Prime m' ∧ (∃ n' : ℕ, a' * b' = n' * n') ∧ a' ≥ 2012 → a' ≥ a) ∧ 
  a = 2025 := by
sorry


end min_value_a_l1007_100738


namespace equal_product_grouping_l1007_100799

theorem equal_product_grouping (numbers : Finset ℕ) 
  (h_numbers : numbers = {12, 30, 42, 44, 57, 91, 95, 143}) :
  (12 * 42 * 95 * 143 : ℕ) = (30 * 44 * 57 * 91 : ℕ) := by
  sorry

end equal_product_grouping_l1007_100799


namespace trajectory_and_line_theorem_l1007_100721

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 49/4
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1/4

-- Define the trajectory of P
def trajectory_P (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = -2

theorem trajectory_and_line_theorem :
  ∃ k : ℝ, k^2 = 2 ∧
  (∀ x y : ℝ, trajectory_P x y →
    (∃ x₁ y₁ x₂ y₂ : ℝ,
      line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
      trajectory_P x₁ y₁ ∧ trajectory_P x₂ y₂ ∧
      dot_product_condition x₁ y₁ x₂ y₂)) :=
sorry

end trajectory_and_line_theorem_l1007_100721


namespace each_child_gets_twenty_cookies_l1007_100701

/-- Represents the cookie distribution problem in Everlee's family -/
def cookie_distribution (total_cookies : ℕ) (num_adults : ℕ) (num_children : ℕ) : ℕ :=
  let adults_share := total_cookies / 3
  let remaining_cookies := total_cookies - adults_share
  remaining_cookies / num_children

/-- Theorem stating that each child gets 20 cookies -/
theorem each_child_gets_twenty_cookies :
  cookie_distribution 120 2 4 = 20 := by
  sorry

end each_child_gets_twenty_cookies_l1007_100701


namespace square_sum_from_means_l1007_100743

theorem square_sum_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 24)
  (h_geometric : Real.sqrt (a * b) = Real.sqrt 156) :
  a^2 + b^2 = 1992 := by
sorry

end square_sum_from_means_l1007_100743


namespace sum_of_powers_of_i_l1007_100761

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i :
  i^255 + i^256 + i^257 + i^258 + i^259 = -i :=
by
  sorry

end sum_of_powers_of_i_l1007_100761


namespace inscribed_rectangle_theorem_l1007_100777

-- Define the triangle
def triangle_sides : (ℝ × ℝ × ℝ) := (10, 17, 21)

-- Define the perimeter of the inscribed rectangle
def rectangle_perimeter : ℝ := 24

-- Define the function to calculate the sides of the inscribed rectangle
def inscribed_rectangle_sides (triangle : ℝ × ℝ × ℝ) (perimeter : ℝ) : (ℝ × ℝ) :=
  sorry

-- Theorem statement
theorem inscribed_rectangle_theorem :
  inscribed_rectangle_sides triangle_sides rectangle_perimeter = (5 + 7/13, 6 + 6/13) :=
sorry

end inscribed_rectangle_theorem_l1007_100777


namespace factorization_3m_squared_minus_6m_l1007_100737

theorem factorization_3m_squared_minus_6m (m : ℝ) : 3 * m^2 - 6 * m = 3 * m * (m - 2) := by
  sorry

end factorization_3m_squared_minus_6m_l1007_100737


namespace matrix_power_4_l1007_100712

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_power_4 : A ^ 4 = !![0, -9; 9, -9] := by sorry

end matrix_power_4_l1007_100712


namespace school_pupils_count_l1007_100741

theorem school_pupils_count (girls : ℕ) (boys : ℕ) (teachers : ℕ) : girls = 308 → boys = 318 → teachers = 36 → girls + boys = 626 := by
  sorry

end school_pupils_count_l1007_100741


namespace treehouse_planks_l1007_100762

/-- The total number of planks Charlie and his father have -/
def total_planks (initial_planks charlie_planks father_planks : ℕ) : ℕ :=
  initial_planks + charlie_planks + father_planks

/-- Theorem stating that the total number of planks is 35 -/
theorem treehouse_planks : total_planks 15 10 10 = 35 := by
  sorry

end treehouse_planks_l1007_100762


namespace tape_shortage_l1007_100728

/-- Proves that 180 feet of tape is insufficient to wrap around a 35x80 foot field and three 5-foot circumference trees, requiring an additional 65 feet. -/
theorem tape_shortage (field_width : ℝ) (field_length : ℝ) (tree_circumference : ℝ) (num_trees : ℕ) (available_tape : ℝ) : 
  field_width = 35 → 
  field_length = 80 → 
  tree_circumference = 5 → 
  num_trees = 3 → 
  available_tape = 180 → 
  (2 * (field_width + field_length) + num_trees * tree_circumference) - available_tape = 65 := by
  sorry

end tape_shortage_l1007_100728


namespace power_inequality_l1007_100792

theorem power_inequality : (1.7 : ℝ) ^ (0.3 : ℝ) > (0.9 : ℝ) ^ (0.3 : ℝ) := by sorry

end power_inequality_l1007_100792


namespace work_completion_equality_prove_new_group_size_l1007_100784

/-- The number of persons in the original group -/
def original_group : ℕ := 15

/-- The number of days the original group takes to complete the work -/
def original_days : ℕ := 18

/-- The fraction of work done by the new group -/
def new_group_work_fraction : ℚ := 1/3

/-- The number of days the new group takes to complete their fraction of work -/
def new_group_days : ℕ := 21

/-- The multiplier for the number of persons in the new group -/
def new_group_multiplier : ℚ := 5/2

/-- The number of persons in the new group -/
def new_group_size : ℕ := 7

theorem work_completion_equality :
  (original_group : ℚ) / original_days = 
  (new_group_multiplier * new_group_size) * new_group_work_fraction / new_group_days :=
by sorry

/-- The main theorem proving the size of the new group -/
theorem prove_new_group_size : 
  ∃ (n : ℕ), n = new_group_size ∧
  (original_group : ℚ) / original_days = 
  (new_group_multiplier * n) * new_group_work_fraction / new_group_days :=
by sorry

end work_completion_equality_prove_new_group_size_l1007_100784


namespace election_votes_total_l1007_100734

theorem election_votes_total (V A B C : ℝ) 
  (h1 : A = B + 0.10 * V)
  (h2 : A = C + 0.15 * V)
  (h3 : A - 3000 = B + 3000)
  (h4 : B + 3000 = A - 0.10 * V)
  (h5 : B + 3000 = C + 0.05 * V) :
  V = 60000 := by
sorry

end election_votes_total_l1007_100734


namespace expression_factorization_l1007_100700

theorem expression_factorization (x : ℝ) : 
  (21 * x^4 + 90 * x^3 + 40 * x - 10) - (7 * x^4 + 6 * x^3 + 8 * x - 6) = 
  2 * x * (7 * x^3 + 42 * x^2 + 16) - 4 := by
  sorry

end expression_factorization_l1007_100700


namespace inequality_solution_set_l1007_100726

def f (x : ℝ) := abs x + abs (x - 4)

theorem inequality_solution_set :
  {x : ℝ | f (x^2 + 2) > f x} = {x | x < -2 ∨ x > Real.sqrt 2} :=
by sorry

end inequality_solution_set_l1007_100726


namespace star_properties_l1007_100716

-- Define the set T of non-zero real numbers
def T : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation ★
def star (x y : ℝ) : ℝ := 3 * x * y + x + y

-- Theorem statement
theorem star_properties :
  (∀ x ∈ T, star x (-1) ≠ x ∨ star (-1) x ≠ x) ∧
  (star 1 (-1/2) = -1 ∧ star (-1/2) 1 = -1) :=
sorry

end star_properties_l1007_100716


namespace complex_modulus_problem_l1007_100714

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I)^2 = 3 - 4 * Complex.I) :
  Complex.abs z = 5 / 2 := by
  sorry

end complex_modulus_problem_l1007_100714


namespace expression_evaluation_l1007_100791

theorem expression_evaluation :
  let f : ℝ → ℝ := λ x => (x^2 - 2*x - 8) / (x - 4)
  f 5 = 7 := by sorry

end expression_evaluation_l1007_100791


namespace ordering_proof_l1007_100783

theorem ordering_proof (a b c : ℝ) 
  (ha : a = Real.log 2.6)
  (hb : b = 0.5 * 1.8^2)
  (hc : c = 1.1^5) : 
  b > c ∧ c > a := by
  sorry

end ordering_proof_l1007_100783


namespace composition_of_even_is_even_l1007_100760

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Theorem statement
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : EvenFunction f) :
  EvenFunction (f ∘ f) := by
sorry

end composition_of_even_is_even_l1007_100760


namespace polynomial_value_theorem_l1007_100754

/-- Given a polynomial f(x) = ax^4 + bx^3 + cx^2 + dx + e where f(-3) = -5,
    prove that 8a - 4b + 2c - d + e = -5 -/
theorem polynomial_value_theorem (a b c d e : ℝ) :
  (fun x : ℝ ↦ a * x^4 + b * x^3 + c * x^2 + d * x + e) (-3) = -5 →
  8 * a - 4 * b + 2 * c - d + e = -5 := by
  sorry

end polynomial_value_theorem_l1007_100754


namespace students_attending_game_l1007_100739

/-- Proves the number of students attending a football game -/
theorem students_attending_game (total_attendees : ℕ) (student_price non_student_price : ℕ) (total_revenue : ℕ) : 
  total_attendees = 3000 →
  student_price = 10 →
  non_student_price = 15 →
  total_revenue = 36250 →
  ∃ (students non_students : ℕ),
    students + non_students = total_attendees ∧
    students * student_price + non_students * non_student_price = total_revenue ∧
    students = 1750 :=
by sorry

end students_attending_game_l1007_100739


namespace polynomial_remainder_l1007_100746

/-- Given a polynomial Q(x) such that Q(17) = 53 and Q(53) = 17,
    the remainder when Q(x) is divided by (x - 17)(x - 53) is -x + 70 -/
theorem polynomial_remainder (Q : ℝ → ℝ) (h1 : Q 17 = 53) (h2 : Q 53 = 17) :
  ∃ (R : ℝ → ℝ), ∀ x, Q x = (x - 17) * (x - 53) * R x + (-x + 70) :=
sorry

end polynomial_remainder_l1007_100746


namespace train_length_proof_l1007_100770

/-- Proves that a train with given speed crossing a bridge of known length in a specific time has a particular length -/
theorem train_length_proof (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) (train_length : ℝ) : 
  bridge_length = 320 →
  crossing_time = 40 →
  train_speed_kmh = 42.3 →
  train_length = 150 →
  (train_length + bridge_length) = (train_speed_kmh * 1000 / 3600) * crossing_time := by
  sorry

#check train_length_proof

end train_length_proof_l1007_100770


namespace product_63_57_l1007_100742

theorem product_63_57 : 63 * 57 = 3591 := by
  sorry

end product_63_57_l1007_100742


namespace basketball_distribution_l1007_100713

theorem basketball_distribution (total_basketballs : ℕ) (basketballs_per_class : ℕ) (num_classes : ℕ) : 
  total_basketballs = 54 → 
  basketballs_per_class = 7 → 
  total_basketballs = num_classes * basketballs_per_class →
  num_classes = 7 := by
sorry

end basketball_distribution_l1007_100713


namespace price_decrease_percentage_l1007_100709

theorem price_decrease_percentage (original_price new_price : ℚ) :
  original_price = 1750 →
  new_price = 1050 →
  (original_price - new_price) / original_price * 100 = 40 := by
  sorry

end price_decrease_percentage_l1007_100709


namespace complex_subtraction_l1007_100795

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + 3*I) :
  a - 3*b = -1 - 12*I := by sorry

end complex_subtraction_l1007_100795


namespace work_completion_proof_l1007_100727

/-- The number of days B takes to finish the work alone -/
def B : ℝ := 10

/-- The number of days A and B work together -/
def together_days : ℝ := 2

/-- The number of days B takes to finish the remaining work after A leaves -/
def B_remaining : ℝ := 3.0000000000000004

/-- The number of days A takes to finish the work alone -/
def A : ℝ := 4

theorem work_completion_proof :
  2 * (1 / A + 1 / B) + B_remaining * (1 / B) = 1 :=
by sorry

end work_completion_proof_l1007_100727


namespace complex_fraction_sum_l1007_100722

theorem complex_fraction_sum (a b : ℝ) :
  (a + b * Complex.I : ℂ) = (3 + Complex.I) / (1 - Complex.I) → a + b = 3 := by
  sorry

end complex_fraction_sum_l1007_100722


namespace simplify_expression_l1007_100756

theorem simplify_expression (x y : ℝ) :
  (25 * x + 70 * y) + (15 * x + 34 * y) - (13 * x + 55 * y) = 27 * x + 49 * y := by
  sorry

end simplify_expression_l1007_100756


namespace ice_cream_combinations_l1007_100708

/-- Represents the number of cone types -/
def num_cone_types : ℕ := 2

/-- Represents the maximum number of scoops -/
def max_scoops : ℕ := 3

/-- Represents the number of ice cream flavors -/
def num_flavors : ℕ := 4

/-- Represents the number of topping choices -/
def num_toppings : ℕ := 4

/-- Represents the maximum number of toppings allowed -/
def max_toppings : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the total number of ice cream combinations -/
def total_combinations : ℕ := 
  let one_scoop := num_flavors
  let two_scoops := num_flavors + choose num_flavors 2
  let three_scoops := num_flavors + num_flavors * (num_flavors - 1) + choose num_flavors 3
  let scoop_combinations := one_scoop + two_scoops + three_scoops
  let topping_combinations := 1 + num_toppings + choose num_toppings 2
  num_cone_types * scoop_combinations * topping_combinations

/-- Theorem stating that the total number of ice cream combinations is 748 -/
theorem ice_cream_combinations : total_combinations = 748 := by sorry

end ice_cream_combinations_l1007_100708
