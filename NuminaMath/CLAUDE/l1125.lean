import Mathlib

namespace NUMINAMATH_CALUDE_sunflower_seed_contest_l1125_112543

/-- The total number of seeds eaten by five players in a sunflower seed eating contest -/
def total_seeds (player1 player2 player3 player4 player5 : ℕ) : ℕ :=
  player1 + player2 + player3 + player4 + player5

/-- Theorem stating the total number of seeds eaten by the five players -/
theorem sunflower_seed_contest : ∃ (player1 player2 player3 player4 player5 : ℕ),
  player1 = 78 ∧
  player2 = 53 ∧
  player3 = player2 + 30 ∧
  player4 = 2 * player3 ∧
  player5 = (player1 + player2 + player3 + player4) / 4 ∧
  total_seeds player1 player2 player3 player4 player5 = 475 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_seed_contest_l1125_112543


namespace NUMINAMATH_CALUDE_porter_painting_sale_l1125_112509

/-- The sale price of Porter's previous painting in dollars -/
def previous_sale : ℕ := 9000

/-- The sale price of Porter's most recent painting in dollars -/
def recent_sale : ℕ := 5 * previous_sale - 1000

theorem porter_painting_sale : recent_sale = 44000 := by
  sorry

end NUMINAMATH_CALUDE_porter_painting_sale_l1125_112509


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1125_112591

theorem quadratic_inequality_solution (b : ℝ) :
  (∀ x, x^2 - 3*x + 6 > 4 ↔ (x < 1 ∨ x > b)) →
  (b = 2 ∧
   ∀ c, 
     (c > 2 → ∀ x, x^2 - (c+2)*x + 2*c < 0 ↔ 2 < x ∧ x < c) ∧
     (c < 2 → ∀ x, x^2 - (c+2)*x + 2*c < 0 ↔ c < x ∧ x < 2) ∧
     (c = 2 → ∀ x, ¬(x^2 - (c+2)*x + 2*c < 0))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1125_112591


namespace NUMINAMATH_CALUDE_cubic_tangent_line_theorem_l1125_112540

/-- Given a cubic function f(x) = ax^3 + bx^2 + cx + d, 
    if the equation of the tangent line to its graph at x=0 is 24x + y - 12 = 0, 
    then c + 2d = 0 -/
theorem cubic_tangent_line_theorem (a b c d : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x^2 + c * x + d
  let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 2 * b * x + c
  (∀ x y, y = f x → (24 * 0 + y - 12 = 0 ↔ 24 * x + y - 12 = 0)) →
  c + 2 * d = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_tangent_line_theorem_l1125_112540


namespace NUMINAMATH_CALUDE_no_outliers_l1125_112566

def data_set : List ℝ := [2, 11, 23, 23, 25, 35, 41, 41, 55, 67, 85]
def Q1 : ℝ := 23
def Q2 : ℝ := 35
def Q3 : ℝ := 55

def is_outlier (x : ℝ) : Prop :=
  let IQR := Q3 - Q1
  x < Q1 - 2 * IQR ∨ x > Q3 + 2 * IQR

theorem no_outliers : ∀ x ∈ data_set, ¬(is_outlier x) := by sorry

end NUMINAMATH_CALUDE_no_outliers_l1125_112566


namespace NUMINAMATH_CALUDE_chocolate_theorem_l1125_112551

/-- The difference between 75% of Robert's chocolates and the total number of chocolates Nickel and Penelope ate -/
def chocolate_difference (robert : ℝ) (nickel : ℝ) (penelope : ℝ) : ℝ :=
  0.75 * robert - (nickel + penelope)

/-- Theorem stating the difference in chocolates -/
theorem chocolate_theorem :
  chocolate_difference 13 4 7.5 = -1.75 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_theorem_l1125_112551


namespace NUMINAMATH_CALUDE_pages_per_side_l1125_112550

/-- Given the conditions of James' printing job, prove the number of pages per side. -/
theorem pages_per_side (num_books : ℕ) (pages_per_book : ℕ) (num_sheets : ℕ) : 
  num_books = 2 → 
  pages_per_book = 600 → 
  num_sheets = 150 → 
  (num_books * pages_per_book) / (num_sheets * 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_side_l1125_112550


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_a_le_e_plus_one_zeros_product_lt_one_l1125_112504

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / x - Real.log x + x - a

theorem f_nonnegative_iff_a_le_e_plus_one (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
sorry

theorem zeros_product_lt_one (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → f a x₁ = 0 → f a x₂ = 0 → x₁ * x₂ < 1 :=
sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_a_le_e_plus_one_zeros_product_lt_one_l1125_112504


namespace NUMINAMATH_CALUDE_min_value_a_squared_minus_b_l1125_112595

/-- The function f(x) = x^4 + ax^3 + bx^2 + ax + 1 -/
def f (a b x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + a*x + 1

/-- Theorem: If f(x) has at least one root, then a^2 - b ≥ 1 -/
theorem min_value_a_squared_minus_b (a b : ℝ) :
  (∃ x, f a b x = 0) → a^2 - b ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_squared_minus_b_l1125_112595


namespace NUMINAMATH_CALUDE_card_difference_l1125_112525

theorem card_difference (heike anton ann : ℕ) : 
  anton = 3 * heike →
  ann = 6 * heike →
  ann = 60 →
  ann - anton = 30 := by
sorry

end NUMINAMATH_CALUDE_card_difference_l1125_112525


namespace NUMINAMATH_CALUDE_carpet_dimensions_l1125_112577

/-- A rectangular carpet with integer side lengths. -/
structure Carpet where
  length : ℕ
  width : ℕ

/-- A rectangular room. -/
structure Room where
  length : ℕ
  width : ℕ

/-- Predicate to check if a carpet fits perfectly (diagonally) in a room. -/
def fits_perfectly (c : Carpet) (r : Room) : Prop :=
  (c.length ^ 2 + c.width ^ 2 : ℕ) = r.length ^ 2 + r.width ^ 2

/-- The main theorem about the carpet dimensions. -/
theorem carpet_dimensions :
  ∀ (c : Carpet) (r1 r2 : Room),
    r1.width = 50 →
    r2.width = 38 →
    r1.length = r2.length →
    fits_perfectly c r1 →
    fits_perfectly c r2 →
    c.length = 50 ∧ c.width = 25 := by
  sorry

end NUMINAMATH_CALUDE_carpet_dimensions_l1125_112577


namespace NUMINAMATH_CALUDE_tom_total_games_l1125_112503

/-- The number of hockey games Tom attended over two years -/
def total_games (games_this_year games_last_year : ℕ) : ℕ :=
  games_this_year + games_last_year

/-- Theorem: Tom attended 13 hockey games in total over two years -/
theorem tom_total_games :
  total_games 4 9 = 13 := by
  sorry

end NUMINAMATH_CALUDE_tom_total_games_l1125_112503


namespace NUMINAMATH_CALUDE_exists_cube_root_of_3_15_l1125_112596

theorem exists_cube_root_of_3_15 : ∃ n : ℕ, 3^12 * 3^3 = n^3 := by
  sorry

end NUMINAMATH_CALUDE_exists_cube_root_of_3_15_l1125_112596


namespace NUMINAMATH_CALUDE_smallest_volume_is_180_l1125_112567

/-- Represents the dimensions and cube counts of a rectangular box. -/
structure BoxDimensions where
  a : ℕ+  -- length
  b : ℕ+  -- width
  c : ℕ+  -- height
  red_in_bc : ℕ+  -- number of red cubes in each 1×b×c layer
  green_in_bc : ℕ+  -- number of green cubes in each 1×b×c layer
  green_in_ac : ℕ+  -- number of green cubes in each a×1×c layer
  yellow_in_ac : ℕ+  -- number of yellow cubes in each a×1×c layer

/-- Checks if the given box dimensions satisfy the problem conditions. -/
def valid_box_dimensions (box : BoxDimensions) : Prop :=
  box.red_in_bc = 9 ∧
  box.green_in_bc = 12 ∧
  box.green_in_ac = 20 ∧
  box.yellow_in_ac = 25

/-- Calculates the volume of the box. -/
def box_volume (box : BoxDimensions) : ℕ :=
  box.a * box.b * box.c

/-- The main theorem stating that the smallest possible volume is 180. -/
theorem smallest_volume_is_180 :
  ∀ box : BoxDimensions, valid_box_dimensions box → box_volume box ≥ 180 :=
by sorry

end NUMINAMATH_CALUDE_smallest_volume_is_180_l1125_112567


namespace NUMINAMATH_CALUDE_tan_two_alpha_l1125_112586

theorem tan_two_alpha (α β : ℝ) (h1 : Real.tan (α - β) = -3/2) (h2 : Real.tan (α + β) = 3) :
  Real.tan (2 * α) = 3/11 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_alpha_l1125_112586


namespace NUMINAMATH_CALUDE_day_after_53_days_l1125_112583

-- Define the days of the week
inductive DayOfWeek
  | Friday
  | Saturday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday

-- Define a function to advance the day by one
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday

-- Define a function to advance the day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Theorem statement
theorem day_after_53_days : advanceDay DayOfWeek.Friday 53 = DayOfWeek.Tuesday := by
  sorry

end NUMINAMATH_CALUDE_day_after_53_days_l1125_112583


namespace NUMINAMATH_CALUDE_salary_calculation_l1125_112556

theorem salary_calculation (S : ℝ) 
  (food_expense : S / 5 = S * (1 / 5))
  (rent_expense : S / 10 = S * (1 / 10))
  (clothes_expense : S * (3 / 5) = S * (3 / 5))
  (remaining : S - (S * (1 / 5) + S * (1 / 10) + S * (3 / 5)) = 16000) :
  S = 160000 := by
  sorry

end NUMINAMATH_CALUDE_salary_calculation_l1125_112556


namespace NUMINAMATH_CALUDE_math_city_intersections_l1125_112581

/-- Represents a city with a given number of streets -/
structure City where
  num_streets : ℕ
  no_parallel : Bool
  no_triple_intersections : Bool

/-- Calculates the number of intersections in a city -/
def num_intersections (c : City) : ℕ :=
  (c.num_streets * (c.num_streets - 1)) / 2

/-- Theorem: A city with 10 streets, no parallel streets, and no triple intersections has 45 intersections -/
theorem math_city_intersections :
  ∀ (c : City), c.num_streets = 10 → c.no_parallel = true → c.no_triple_intersections = true →
  num_intersections c = 45 :=
by sorry

end NUMINAMATH_CALUDE_math_city_intersections_l1125_112581


namespace NUMINAMATH_CALUDE_tonya_stamps_left_l1125_112529

/-- Represents the trade of matchbooks for stamps --/
def stamps_after_trade (initial_stamps : ℕ) (matchbooks : ℕ) : ℕ :=
  let matches_per_book : ℕ := 24
  let matches_per_stamp : ℕ := 12
  let total_matches : ℕ := matchbooks * matches_per_book
  let stamps_traded : ℕ := total_matches / matches_per_stamp
  initial_stamps - stamps_traded

/-- Theorem stating that Tonya will have 3 stamps left after the trade --/
theorem tonya_stamps_left : stamps_after_trade 13 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tonya_stamps_left_l1125_112529


namespace NUMINAMATH_CALUDE_total_cost_of_bottle_caps_l1125_112568

/-- The cost of a single bottle cap in dollars -/
def bottle_cap_cost : ℕ := 2

/-- The number of bottle caps -/
def num_bottle_caps : ℕ := 6

/-- Theorem: The total cost of 6 bottle caps is $12 -/
theorem total_cost_of_bottle_caps : 
  bottle_cap_cost * num_bottle_caps = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_bottle_caps_l1125_112568


namespace NUMINAMATH_CALUDE_complex_division_equality_l1125_112562

theorem complex_division_equality : (2 - I) / (2 + I) = 3/5 - 4/5 * I := by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l1125_112562


namespace NUMINAMATH_CALUDE_log_five_negative_one_l1125_112512

theorem log_five_negative_one (x : ℝ) (h1 : x > 0) (h2 : Real.log x / Real.log 5 = -1) : x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_log_five_negative_one_l1125_112512


namespace NUMINAMATH_CALUDE_dice_probability_l1125_112563

def red_die := Finset.range 6
def blue_die := Finset.range 6

def event_M (x : ℕ) : Prop := x % 3 = 0 ∧ x ∈ red_die
def event_N (x y : ℕ) : Prop := x + y > 8 ∧ x ∈ red_die ∧ y ∈ blue_die

def P_MN : ℚ := 5 / 36
def P_M : ℚ := 1 / 3

theorem dice_probability : (P_MN / P_M) = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l1125_112563


namespace NUMINAMATH_CALUDE_billiard_path_equals_diagonals_l1125_112510

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a point on the rectangle's perimeter -/
structure PerimeterPoint where
  x : ℝ
  y : ℝ

/-- Calculates the length of the billiard ball's path -/
def billiardPathLength (rect : Rectangle) (start : PerimeterPoint) : ℝ :=
  sorry

/-- Calculates the sum of the diagonals of the rectangle -/
def sumOfDiagonals (rect : Rectangle) : ℝ :=
  sorry

/-- Theorem: The billiard path length equals the sum of the rectangle's diagonals -/
theorem billiard_path_equals_diagonals (rect : Rectangle) (start : PerimeterPoint) :
  billiardPathLength rect start = sumOfDiagonals rect :=
  sorry

end NUMINAMATH_CALUDE_billiard_path_equals_diagonals_l1125_112510


namespace NUMINAMATH_CALUDE_g_composition_of_3_l1125_112552

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 2*n + 1 else 4*n - 3

theorem g_composition_of_3 : g (g (g 3)) = 241 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_3_l1125_112552


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1125_112539

theorem line_tangent_to_parabola :
  ∃! (x y : ℝ), 4 * x + 3 * y + 18 = 0 ∧ y^2 = 32 * x := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1125_112539


namespace NUMINAMATH_CALUDE_jelly_bean_color_match_probability_l1125_112558

def claire_green : ℕ := 2
def claire_red : ℕ := 2
def daniel_green : ℕ := 2
def daniel_yellow : ℕ := 3
def daniel_red : ℕ := 4

def claire_total : ℕ := claire_green + claire_red
def daniel_total : ℕ := daniel_green + daniel_yellow + daniel_red

theorem jelly_bean_color_match_probability :
  (claire_green / claire_total : ℚ) * (daniel_green / daniel_total : ℚ) +
  (claire_red / claire_total : ℚ) * (daniel_red / daniel_total : ℚ) =
  1 / 3 :=
sorry

end NUMINAMATH_CALUDE_jelly_bean_color_match_probability_l1125_112558


namespace NUMINAMATH_CALUDE_lollipop_distribution_l1125_112521

theorem lollipop_distribution (raspberry mint orange cotton_candy : ℕ) 
  (h1 : raspberry = 60) 
  (h2 : mint = 135) 
  (h3 : orange = 5) 
  (h4 : cotton_candy = 330) 
  (friends : ℕ) 
  (h5 : friends = 15) : 
  (raspberry + mint + orange + cotton_candy) % friends = 5 := by
sorry

end NUMINAMATH_CALUDE_lollipop_distribution_l1125_112521


namespace NUMINAMATH_CALUDE_inequality_constraint_l1125_112574

theorem inequality_constraint (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → x^2 + a*x + 9 ≥ 0) ↔ a ≥ -6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_constraint_l1125_112574


namespace NUMINAMATH_CALUDE_tangential_quadrilateral_theorem_l1125_112580

/-- A tangential quadrilateral with circumscribed and inscribed circles -/
structure TangentialQuadrilateral where
  /-- The radius of the circumscribed circle -/
  R : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The distance between the centers of the circumscribed and inscribed circles -/
  d : ℝ
  /-- R is positive -/
  R_pos : R > 0
  /-- r is positive -/
  r_pos : r > 0
  /-- d is non-negative and less than R -/
  d_bounds : 0 ≤ d ∧ d < R

/-- The main theorem about the relationship between R, r, and d in a tangential quadrilateral -/
theorem tangential_quadrilateral_theorem (q : TangentialQuadrilateral) :
  1 / (q.R + q.d)^2 + 1 / (q.R - q.d)^2 = 1 / q.r^2 := by
  sorry

end NUMINAMATH_CALUDE_tangential_quadrilateral_theorem_l1125_112580


namespace NUMINAMATH_CALUDE_multiple_properties_l1125_112555

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ 
  (∃ p : ℤ, a + b = 4 * p) ∧ 
  (∃ q : ℤ, a + b = 2 * q) := by
sorry

end NUMINAMATH_CALUDE_multiple_properties_l1125_112555


namespace NUMINAMATH_CALUDE_roots_equation_l1125_112584

theorem roots_equation (n r s c d : ℝ) : 
  (c^2 - n*c + 6 = 0) →
  (d^2 - n*d + 6 = 0) →
  ((c^2 + 1/d)^2 - r*(c^2 + 1/d) + s = 0) →
  ((d^2 + 1/c)^2 - r*(d^2 + 1/c) + s = 0) →
  s = n + 217/6 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_l1125_112584


namespace NUMINAMATH_CALUDE_even_swaps_not_restore_order_l1125_112533

/-- A permutation of integers from 1 to n -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The identity permutation (ascending order) -/
def id_perm (n : ℕ) : Permutation n := fun i => i

/-- Swap two elements in a permutation -/
def swap (p : Permutation n) (i j : Fin n) : Permutation n :=
  fun k => if k = i then p j else if k = j then p i else p k

/-- Apply a sequence of swaps to a permutation -/
def apply_swaps (p : Permutation n) (swaps : List (Fin n × Fin n)) : Permutation n :=
  swaps.foldl (fun p' (i, j) => swap p' i j) p

/-- The main theorem -/
theorem even_swaps_not_restore_order (n : ℕ) (swaps : List (Fin n × Fin n)) :
  swaps.length % 2 = 0 → apply_swaps (id_perm n) swaps ≠ id_perm n :=
sorry

end NUMINAMATH_CALUDE_even_swaps_not_restore_order_l1125_112533


namespace NUMINAMATH_CALUDE_triangle_perimeter_max_l1125_112535

open Real

theorem triangle_perimeter_max (x : ℝ) (h : 0 < x ∧ x < 2 * π / 3) :
  let y := 4 * Real.sqrt 3 * sin (x + π / 6) + 2 * Real.sqrt 3
  ∃ (y_max : ℝ), y ≤ y_max ∧ y_max = 6 * Real.sqrt 3 := by
  sorry

#check triangle_perimeter_max

end NUMINAMATH_CALUDE_triangle_perimeter_max_l1125_112535


namespace NUMINAMATH_CALUDE_thirteen_binary_l1125_112532

/-- Converts a natural number to its binary representation as a list of booleans -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Checks if a list of booleans represents a given natural number in binary -/
def is_binary_rep (n : ℕ) (l : List Bool) : Prop :=
  to_binary n = l.reverse

theorem thirteen_binary :
  is_binary_rep 13 [true, false, true, true] := by sorry

end NUMINAMATH_CALUDE_thirteen_binary_l1125_112532


namespace NUMINAMATH_CALUDE_percentage_equality_l1125_112594

theorem percentage_equality (x : ℝ) (h : 0.3 * (0.4 * x) = 60) : 0.4 * (0.3 * x) = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l1125_112594


namespace NUMINAMATH_CALUDE_gcd_111_1850_l1125_112530

theorem gcd_111_1850 : Nat.gcd 111 1850 = 37 := by sorry

end NUMINAMATH_CALUDE_gcd_111_1850_l1125_112530


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l1125_112585

-- Define the points
def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (4, 8)
def C : ℝ → ℝ × ℝ := λ x => (5, x)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r.1 - p.1 = t * (q.1 - p.1) ∧ r.2 - p.2 = t * (q.2 - p.2)

-- Theorem statement
theorem collinear_points_x_value :
  ∀ x : ℝ, collinear A B (C x) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_x_value_l1125_112585


namespace NUMINAMATH_CALUDE_sumata_vacation_l1125_112597

/-- The Sumata family vacation problem -/
theorem sumata_vacation (miles_per_day : ℕ) (total_miles : ℕ) (h1 : miles_per_day = 250) (h2 : total_miles = 1250) :
  total_miles / miles_per_day = 5 := by
  sorry

end NUMINAMATH_CALUDE_sumata_vacation_l1125_112597


namespace NUMINAMATH_CALUDE_divisible_by_nine_l1125_112536

theorem divisible_by_nine (n : ℕ) : ∃ k : ℤ, (4 : ℤ)^n + 15*n - 1 = 9*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l1125_112536


namespace NUMINAMATH_CALUDE_population_size_l1125_112501

/-- Given a population with specific birth and death rates, prove the initial population size. -/
theorem population_size (birth_rate death_rate net_growth_rate : ℚ) : 
  birth_rate = 32 →
  death_rate = 11 →
  net_growth_rate = 21 / 1000 →
  (birth_rate - death_rate) / 1000 = net_growth_rate →
  1000 = (birth_rate - death_rate) / net_growth_rate :=
by sorry

end NUMINAMATH_CALUDE_population_size_l1125_112501


namespace NUMINAMATH_CALUDE_divisibility_by_primes_less_than_1966_l1125_112531

theorem divisibility_by_primes_less_than_1966 (n : ℕ) (p : ℕ) (hp : Prime p) (hp_bound : p < 1966) :
  p ∣ (List.range 1966).foldl (λ acc i => acc * ((i + 1) * n + 1)) n :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_primes_less_than_1966_l1125_112531


namespace NUMINAMATH_CALUDE_angle_C_is_30_degrees_l1125_112588

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  3 * Real.sin t.A + 4 * Real.cos t.B = 6

def condition2 (t : Triangle) : Prop :=
  4 * Real.sin t.B + 3 * Real.cos t.A = 1

-- Theorem statement
theorem angle_C_is_30_degrees (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) : 
  t.C = Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_angle_C_is_30_degrees_l1125_112588


namespace NUMINAMATH_CALUDE_base_conversion_512_to_base_7_l1125_112538

theorem base_conversion_512_to_base_7 :
  (1 * 7^3 + 3 * 7^2 + 3 * 7^1 + 1 * 7^0) = 512 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_512_to_base_7_l1125_112538


namespace NUMINAMATH_CALUDE_diff_suit_prob_is_13_17_l1125_112545

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- The suits in a standard deck -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- A function that assigns a suit to each card in the deck -/
def card_suit : Fin 52 → Suit := sorry

/-- The probability of picking two cards of different suits -/
def diff_suit_prob (d : Deck) : ℚ :=
  (39 : ℚ) / 51

/-- Theorem stating that the probability of picking two cards of different suits is 13/17 -/
theorem diff_suit_prob_is_13_17 (d : Deck) :
  diff_suit_prob d = 13 / 17 := by
  sorry

end NUMINAMATH_CALUDE_diff_suit_prob_is_13_17_l1125_112545


namespace NUMINAMATH_CALUDE_soil_bags_needed_l1125_112548

/-- Calculates the number of soil bags needed for raised beds -/
theorem soil_bags_needed
  (num_beds : ℕ)
  (length width height : ℝ)
  (soil_per_bag : ℝ)
  (h_num_beds : num_beds = 2)
  (h_length : length = 8)
  (h_width : width = 4)
  (h_height : height = 1)
  (h_soil_per_bag : soil_per_bag = 4) :
  ⌈(num_beds * length * width * height) / soil_per_bag⌉ = 16 :=
by sorry

end NUMINAMATH_CALUDE_soil_bags_needed_l1125_112548


namespace NUMINAMATH_CALUDE_solve_for_m_l1125_112579

theorem solve_for_m (x y m : ℝ) : 
  x = 2 → 
  y = m → 
  3 * x + 2 * y = 10 → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_m_l1125_112579


namespace NUMINAMATH_CALUDE_two_digit_number_twice_product_of_digits_l1125_112514

theorem two_digit_number_twice_product_of_digits : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ n = 2 * (n / 10) * (n % 10) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_two_digit_number_twice_product_of_digits_l1125_112514


namespace NUMINAMATH_CALUDE_income_a_is_4000_l1125_112571

/-- Represents the financial situation of two individuals A and B -/
structure FinancialSituation where
  incomeRatio : Rat
  expenditureRatio : Rat
  savings : ℕ

/-- Calculates the income of individual A given the financial situation -/
def incomeA (fs : FinancialSituation) : ℕ := sorry

/-- Theorem stating that given the specific financial situation, the income of A is $4000 -/
theorem income_a_is_4000 :
  let fs : FinancialSituation := {
    incomeRatio := 5 / 4,
    expenditureRatio := 3 / 2,
    savings := 1600
  }
  incomeA fs = 4000 := by sorry

end NUMINAMATH_CALUDE_income_a_is_4000_l1125_112571


namespace NUMINAMATH_CALUDE_sine_graph_shift_l1125_112515

theorem sine_graph_shift (x : ℝ) : 
  2 * Real.sin (3 * (x - π/15) + π/5) = 2 * Real.sin (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_sine_graph_shift_l1125_112515


namespace NUMINAMATH_CALUDE_f_positive_iff_a_in_range_l1125_112553

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.log (1 / (a * x + a)) - a

theorem f_positive_iff_a_in_range (a : ℝ) :
  (a > 0 ∧ ∀ x, f a x > 0) ↔ (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_f_positive_iff_a_in_range_l1125_112553


namespace NUMINAMATH_CALUDE_jo_bob_balloon_ride_max_height_l1125_112565

/-- Represents the state of a hot air balloon ride -/
structure BalloonRide where
  ascent_rate : ℝ  -- Rate of ascent when chain is pulled (feet per minute)
  descent_rate : ℝ  -- Rate of descent when chain is released (feet per minute)
  first_pull_duration : ℝ  -- Duration of first chain pull (minutes)
  release_duration : ℝ  -- Duration of chain release (minutes)
  second_pull_duration : ℝ  -- Duration of second chain pull (minutes)

/-- Calculates the maximum height reached during a balloon ride -/
def max_height (ride : BalloonRide) : ℝ :=
  (ride.ascent_rate * ride.first_pull_duration) -
  (ride.descent_rate * ride.release_duration) +
  (ride.ascent_rate * ride.second_pull_duration)

/-- Theorem stating the maximum height reached during Jo-Bob's balloon ride -/
theorem jo_bob_balloon_ride_max_height :
  let ride : BalloonRide := {
    ascent_rate := 50,
    descent_rate := 10,
    first_pull_duration := 15,
    release_duration := 10,
    second_pull_duration := 15
  }
  max_height ride = 1400 := by sorry

end NUMINAMATH_CALUDE_jo_bob_balloon_ride_max_height_l1125_112565


namespace NUMINAMATH_CALUDE_function_periodicity_l1125_112590

def is_periodic (f : ℕ → ℕ) : Prop :=
  ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, f (n + p) = f n

theorem function_periodicity 
  (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, f (n + f n) = f n) 
  (h2 : Set.Finite (Set.range f)) : 
  is_periodic f := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l1125_112590


namespace NUMINAMATH_CALUDE_angle_triple_complement_l1125_112508

theorem angle_triple_complement : ∃ x : ℝ, x + (180 - x) = 180 ∧ x = 3 * (180 - x) ∧ x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l1125_112508


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_twelve_l1125_112524

theorem last_digit_of_one_over_two_to_twelve (n : ℕ) : 
  n = 12 → (1 : ℚ) / (2^n : ℚ) * 10^n % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_twelve_l1125_112524


namespace NUMINAMATH_CALUDE_root_equation_k_value_l1125_112500

theorem root_equation_k_value (k : ℝ) : 
  (∃ x : ℝ, x^2 - k*x - 6 = 0 ∧ x = 3) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_k_value_l1125_112500


namespace NUMINAMATH_CALUDE_farm_milk_production_l1125_112519

theorem farm_milk_production
  (a b c d e : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hd : d > 0)
  (he : e > 0)
  (group_a_production : b = (a * c) * (b / (a * c)))
  (group_b_efficiency : ℝ)
  (hefficiency : group_b_efficiency = 1.2)
  : (group_b_efficiency * b * d * e) / (a * c) = (1.2 * b * d * e) / (a * c) :=
by sorry

end NUMINAMATH_CALUDE_farm_milk_production_l1125_112519


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1125_112560

theorem exam_maximum_marks :
  ∀ (max_marks : ℕ) (paul_marks : ℕ) (failing_margin : ℕ),
    paul_marks = 50 →
    failing_margin = 10 →
    paul_marks + failing_margin = max_marks / 2 →
    max_marks = 120 := by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l1125_112560


namespace NUMINAMATH_CALUDE_not_necessary_nor_sufficient_condition_l1125_112547

theorem not_necessary_nor_sufficient_condition (m n : ℕ+) :
  ¬(∀ a b : ℝ, (a^m.val - b^m.val) * (a^n.val - b^n.val) > 0 → a > b) ∧
  ¬(∀ a b : ℝ, a > b → (a^m.val - b^m.val) * (a^n.val - b^n.val) > 0) :=
by sorry

end NUMINAMATH_CALUDE_not_necessary_nor_sufficient_condition_l1125_112547


namespace NUMINAMATH_CALUDE_root_relation_implies_a_value_l1125_112557

theorem root_relation_implies_a_value (m : ℝ) (h : m > 0) :
  ∃ (a : ℝ), ∀ (x : ℂ),
    (x^4 + 2*x^2 + 1) / (2*(x^3 + x)) = a →
    (∃ (y : ℂ), (x^4 + 2*x^2 + 1) / (2*(x^3 + x)) = a ∧ x = m * y) →
    a = (m + 1) / (2 * m) * Real.sqrt m :=
by sorry

end NUMINAMATH_CALUDE_root_relation_implies_a_value_l1125_112557


namespace NUMINAMATH_CALUDE_sodium_carbonate_mass_fraction_l1125_112520

/-- Given the number of moles of Na₂CO₃, its molar mass, and the total mass of the solution,
    prove that the mass fraction of Na₂CO₃ in the solution is 10%. -/
theorem sodium_carbonate_mass_fraction 
  (n : Real) 
  (M : Real) 
  (m_solution : Real) 
  (h1 : n = 0.125) 
  (h2 : M = 106) 
  (h3 : m_solution = 132.5) : 
  (n * M * 100 / m_solution) = 10 := by
  sorry

#check sodium_carbonate_mass_fraction

end NUMINAMATH_CALUDE_sodium_carbonate_mass_fraction_l1125_112520


namespace NUMINAMATH_CALUDE_runners_meeting_time_l1125_112549

def lap_time_bob : ℕ := 8
def lap_time_carol : ℕ := 9
def lap_time_ted : ℕ := 10

def meeting_time : ℕ := 360

theorem runners_meeting_time :
  Nat.lcm (Nat.lcm lap_time_bob lap_time_carol) lap_time_ted = meeting_time :=
by sorry

end NUMINAMATH_CALUDE_runners_meeting_time_l1125_112549


namespace NUMINAMATH_CALUDE_f_values_l1125_112523

noncomputable def f (x : ℝ) : ℝ :=
  if -1 < x ∧ x < 0 then Real.sin (Real.pi * x^2)
  else if x ≥ 0 then Real.exp (x - 1)
  else 0  -- undefined for x ≤ -1

theorem f_values (a : ℝ) : f 1 + f a = 2 → a = 1 ∨ a = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_values_l1125_112523


namespace NUMINAMATH_CALUDE_cosine_problem_l1125_112537

theorem cosine_problem (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos (α + β) = 12/13) (h4 : Real.cos (2*α + β) = 3/5) :
  Real.cos α = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_cosine_problem_l1125_112537


namespace NUMINAMATH_CALUDE_true_propositions_l1125_112598

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x > y → -x < -y

-- Define proposition q
def q : Prop := ∀ x y : ℝ, x < y → x^2 > y^2

-- Define the four compound propositions
def prop1 : Prop := p ∧ q
def prop2 : Prop := p ∨ q
def prop3 : Prop := p ∧ (¬q)
def prop4 : Prop := (¬p) ∨ q

-- Theorem stating which propositions are true
theorem true_propositions : prop2 ∧ prop3 ∧ ¬prop1 ∧ ¬prop4 := by
  sorry

end NUMINAMATH_CALUDE_true_propositions_l1125_112598


namespace NUMINAMATH_CALUDE_original_curve_equation_l1125_112505

/-- Given a scaling transformation and the equation of the transformed curve,
    prove the equation of the original curve. -/
theorem original_curve_equation
  (x y x' y' : ℝ) -- Real variables for coordinates
  (h1 : x' = 5 * x) -- Scaling transformation for x
  (h2 : y' = 3 * y) -- Scaling transformation for y
  (h3 : 2 * x' ^ 2 + 8 * y' ^ 2 = 1) -- Equation of transformed curve
  : 50 * x ^ 2 + 72 * y ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_original_curve_equation_l1125_112505


namespace NUMINAMATH_CALUDE_digit_product_puzzle_l1125_112582

theorem digit_product_puzzle :
  ∀ (A B C D : Nat),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
    (10 * A + B) * (10 * C + B) = 111 * D →
    10 * A + B < 10 * C + B →
    A = 2 ∧ B = 7 ∧ C = 3 ∧ D = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_product_puzzle_l1125_112582


namespace NUMINAMATH_CALUDE_maxwells_speed_l1125_112511

/-- Proves that Maxwell's walking speed is 4 km/h given the problem conditions -/
theorem maxwells_speed (total_distance : ℝ) (brad_speed : ℝ) (maxwell_time : ℝ) (brad_time : ℝ) 
  (h1 : total_distance = 14)
  (h2 : brad_speed = 6)
  (h3 : maxwell_time = 2)
  (h4 : brad_time = 1)
  (h5 : maxwell_time * maxwell_speed + brad_time * brad_speed = total_distance) :
  maxwell_speed = 4 := by
  sorry

#check maxwells_speed

end NUMINAMATH_CALUDE_maxwells_speed_l1125_112511


namespace NUMINAMATH_CALUDE_problem_statement_l1125_112576

theorem problem_statement : (2112 - 2021)^2 / 169 = 49 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1125_112576


namespace NUMINAMATH_CALUDE_percentage_of_students_passed_l1125_112593

/-- The percentage of students who passed an examination, given the total number of students and the number of students who failed. -/
theorem percentage_of_students_passed (total : ℕ) (failed : ℕ) (h1 : total = 840) (h2 : failed = 546) :
  (((total - failed : ℚ) / total) * 100 : ℚ) = 35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_students_passed_l1125_112593


namespace NUMINAMATH_CALUDE_thirtieth_triangular_and_sum_l1125_112564

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem thirtieth_triangular_and_sum :
  (triangular_number 30 = 465) ∧
  (triangular_number 30 + triangular_number 31 = 961) := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_and_sum_l1125_112564


namespace NUMINAMATH_CALUDE_triangle_max_area_l1125_112526

/-- Given a triangle ABC with circumradius 1 and tan(A) / tan(B) = (2c - b) / b, 
    the maximum area of the triangle is 3√3 / 4 -/
theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a = 2 * Real.sin A ∧
  b = 2 * Real.sin B ∧
  c = 2 * Real.sin C ∧
  Real.tan A / Real.tan B = (2 * c - b) / b →
  (∃ (S : ℝ), S = 1/2 * b * c * Real.sin A ∧ 
    ∀ (S' : ℝ), S' = 1/2 * b * c * Real.sin A → S' ≤ 3 * Real.sqrt 3 / 4) :=
by sorry


end NUMINAMATH_CALUDE_triangle_max_area_l1125_112526


namespace NUMINAMATH_CALUDE_river_current_speed_l1125_112513

/-- Proves that given a ship with a maximum speed of 20 km/h in still water,
    if it takes the same time to travel 100 km downstream as it does to travel 60 km upstream,
    then the speed of the river current is 5 km/h. -/
theorem river_current_speed :
  let ship_speed : ℝ := 20
  let downstream_distance : ℝ := 100
  let upstream_distance : ℝ := 60
  ∀ current_speed : ℝ,
    (downstream_distance / (ship_speed + current_speed) = upstream_distance / (ship_speed - current_speed)) →
    current_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_river_current_speed_l1125_112513


namespace NUMINAMATH_CALUDE_positive_naturals_less_than_three_eq_one_two_l1125_112518

def positive_naturals_less_than_three : Set ℕ := {x | x > 0 ∧ x < 3}

theorem positive_naturals_less_than_three_eq_one_two : 
  positive_naturals_less_than_three = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_positive_naturals_less_than_three_eq_one_two_l1125_112518


namespace NUMINAMATH_CALUDE_total_scheduling_arrangements_l1125_112522

/-- Represents the total number of periods in a day -/
def total_periods : ℕ := 6

/-- Represents the number of morning periods -/
def morning_periods : ℕ := 4

/-- Represents the number of afternoon periods -/
def afternoon_periods : ℕ := 2

/-- Represents the total number of subjects to be scheduled -/
def total_subjects : ℕ := 6

/-- Represents the number of ways to schedule Math in the morning -/
def math_morning_options : ℕ := 4

/-- Represents the number of ways to schedule Physical Education (excluding first morning period) -/
def pe_options : ℕ := 5

/-- Represents the number of ways to arrange the remaining subjects -/
def remaining_arrangements : ℕ := 24

/-- Theorem stating the total number of different scheduling arrangements -/
theorem total_scheduling_arrangements :
  math_morning_options * pe_options * remaining_arrangements = 480 := by
  sorry

end NUMINAMATH_CALUDE_total_scheduling_arrangements_l1125_112522


namespace NUMINAMATH_CALUDE_jane_stopped_babysitting_16_years_ago_l1125_112599

/-- Represents a person with their current age and the age they started babysitting -/
structure Babysitter where
  current_age : ℕ
  start_age : ℕ

/-- Represents a person who was babysat -/
structure BabysatPerson where
  current_age : ℕ

def Babysitter.max_babysat_age (b : Babysitter) : ℕ := b.current_age / 2

def years_since_stopped_babysitting (b : Babysitter) (p : BabysatPerson) : ℕ :=
  b.current_age - p.current_age

theorem jane_stopped_babysitting_16_years_ago
  (jane : Babysitter)
  (oldest_babysat : BabysatPerson)
  (h1 : jane.current_age = 32)
  (h2 : jane.start_age = 16)
  (h3 : oldest_babysat.current_age = 24)
  (h4 : oldest_babysat.current_age ≤ jane.max_babysat_age) :
  years_since_stopped_babysitting jane oldest_babysat = 16 := by
  sorry

#check jane_stopped_babysitting_16_years_ago

end NUMINAMATH_CALUDE_jane_stopped_babysitting_16_years_ago_l1125_112599


namespace NUMINAMATH_CALUDE_polygon_with_720_degree_sum_is_hexagon_l1125_112502

/-- A polygon with interior angles summing to 720° has 6 sides -/
theorem polygon_with_720_degree_sum_is_hexagon :
  ∀ (n : ℕ), (n - 2) * 180 = 720 → n = 6 :=
by sorry

end NUMINAMATH_CALUDE_polygon_with_720_degree_sum_is_hexagon_l1125_112502


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l1125_112541

def motion_equation (t : ℝ) : ℝ := 2 * t^2 + 3

theorem instantaneous_velocity_at_2 :
  (deriv motion_equation) 2 = 8 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l1125_112541


namespace NUMINAMATH_CALUDE_variations_formula_l1125_112507

/-- The number of r-class variations from n elements where the first s elements occur -/
def variations (n r s : ℕ) : ℕ :=
  (Nat.factorial (n - s) * Nat.factorial r) / (Nat.factorial (r - s) * Nat.factorial (n - r))

/-- Theorem stating the number of r-class variations from n elements where the first s elements occur -/
theorem variations_formula (n r s : ℕ) (h1 : s < r) (h2 : r ≤ n) :
  variations n r s = (Nat.factorial (n - s) * Nat.factorial r) / (Nat.factorial (r - s) * Nat.factorial (n - r)) :=
by sorry

end NUMINAMATH_CALUDE_variations_formula_l1125_112507


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l1125_112572

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 62 :=
sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l1125_112572


namespace NUMINAMATH_CALUDE_cucumbers_per_kind_paulines_garden_cucumbers_l1125_112546

/-- Calculates the number of cucumbers of each kind in Pauline's garden. -/
theorem cucumbers_per_kind (total_spaces : ℕ) (total_tomatoes : ℕ) (total_potatoes : ℕ) 
  (cucumber_kinds : ℕ) (empty_spaces : ℕ) : ℕ :=
  by
  have filled_spaces : ℕ := total_spaces - empty_spaces
  have non_cucumber_spaces : ℕ := total_tomatoes + total_potatoes
  have cucumber_spaces : ℕ := filled_spaces - non_cucumber_spaces
  exact cucumber_spaces / cucumber_kinds

/-- Proves that Pauline has planted 4 cucumbers of each kind in her garden. -/
theorem paulines_garden_cucumbers : 
  cucumbers_per_kind 150 15 30 5 85 = 4 :=
by sorry

end NUMINAMATH_CALUDE_cucumbers_per_kind_paulines_garden_cucumbers_l1125_112546


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l1125_112578

theorem polygon_sides_from_angle_sum (n : ℕ) (sum_angles : ℝ) : 
  sum_angles = 900 → (n - 2) * 180 = sum_angles → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l1125_112578


namespace NUMINAMATH_CALUDE_sector_radius_l1125_112570

/-- Given a sector of a circle with perimeter 144 cm and central angle π/3 radians,
    prove that the radius of the circle is 432 / (6 + π) cm. -/
theorem sector_radius (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 144) (h2 : central_angle = π/3) :
  ∃ r : ℝ, r = 432 / (6 + π) ∧ perimeter = 2*r + r * central_angle := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l1125_112570


namespace NUMINAMATH_CALUDE_cafeteria_line_swaps_l1125_112544

/-- Represents a student in the line -/
inductive Student
| Boy : Student
| Girl : Student

/-- The initial line of students -/
def initial_line : List Student :=
  (List.range 8).bind (fun _ => [Student.Boy, Student.Girl])

/-- The final line of students -/
def final_line : List Student :=
  (List.replicate 8 Student.Girl) ++ (List.replicate 8 Student.Boy)

/-- The number of swaps required -/
def num_swaps : Nat := (List.range 8).sum

theorem cafeteria_line_swaps :
  num_swaps = 36 ∧
  num_swaps = (initial_line.length / 2) * ((initial_line.length / 2) + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_cafeteria_line_swaps_l1125_112544


namespace NUMINAMATH_CALUDE_scientific_notation_of_2310000_l1125_112534

/-- Proves that 2,310,000 is equal to 2.31 × 10^6 in scientific notation -/
theorem scientific_notation_of_2310000 : 
  2310000 = 2.31 * (10 : ℝ)^6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2310000_l1125_112534


namespace NUMINAMATH_CALUDE_base5_division_theorem_l1125_112559

/-- Converts a number from base 5 to decimal --/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from decimal to base 5 --/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base5_division_theorem :
  let dividend := [1, 3, 4, 2]  -- 2431₅ in reverse order
  let divisor := [3, 2]         -- 23₅ in reverse order
  let quotient := [3, 0, 1]     -- 103₅ in reverse order
  (base5ToDecimal dividend) / (base5ToDecimal divisor) = base5ToDecimal quotient :=
by sorry

end NUMINAMATH_CALUDE_base5_division_theorem_l1125_112559


namespace NUMINAMATH_CALUDE_star_six_five_l1125_112575

-- Define the star operation
def star (a b : ℕ+) : ℚ :=
  (a.val * (2 * b.val)) / (a.val + 2 * b.val + 3)

-- Theorem statement
theorem star_six_five :
  star 6 5 = 60 / 19 := by
  sorry

end NUMINAMATH_CALUDE_star_six_five_l1125_112575


namespace NUMINAMATH_CALUDE_cheerleader_group_composition_l1125_112561

theorem cheerleader_group_composition :
  let total_males : ℕ := 10
  let males_chose_malt : ℕ := 6
  let females_chose_malt : ℕ := 8
  let total_chose_malt : ℕ := males_chose_malt + females_chose_malt
  let total_chose_coke : ℕ := total_chose_malt / 2
  let females_chose_coke : ℕ := total_chose_coke
  total_males = 10 →
  males_chose_malt = 6 →
  females_chose_malt = 8 →
  total_chose_malt = 2 * total_chose_coke →
  (females_chose_malt + females_chose_coke : ℕ) = 15
  := by sorry

end NUMINAMATH_CALUDE_cheerleader_group_composition_l1125_112561


namespace NUMINAMATH_CALUDE_shortest_rope_length_l1125_112528

theorem shortest_rope_length (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (ratio : b = (5/4) * a ∧ c = (6/4) * a) (sum_condition : a + c = b + 100) : 
  a = 80 := by
sorry

end NUMINAMATH_CALUDE_shortest_rope_length_l1125_112528


namespace NUMINAMATH_CALUDE_middle_of_five_consecutive_sum_60_l1125_112592

theorem middle_of_five_consecutive_sum_60 (a b c d e : ℕ) : 
  (a + b + c + d + e = 60) → 
  (b = a + 1) → 
  (c = b + 1) → 
  (d = c + 1) → 
  (e = d + 1) → 
  c = 12 := by
sorry

end NUMINAMATH_CALUDE_middle_of_five_consecutive_sum_60_l1125_112592


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l1125_112516

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

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l1125_112516


namespace NUMINAMATH_CALUDE_max_parts_three_planes_l1125_112573

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- We don't need to define the specifics of a plane for this statement

/-- The number of parts that a set of planes divides 3D space into -/
def num_parts (planes : List Plane3D) : ℕ :=
  sorry -- Definition would go here

/-- The maximum number of parts that three planes can divide 3D space into -/
theorem max_parts_three_planes :
  ∃ (planes : List Plane3D), planes.length = 3 ∧ 
  ∀ (other_planes : List Plane3D), other_planes.length = 3 →
  num_parts other_planes ≤ num_parts planes ∧ num_parts planes = 8 :=
sorry

end NUMINAMATH_CALUDE_max_parts_three_planes_l1125_112573


namespace NUMINAMATH_CALUDE_ellipse_k_value_l1125_112517

/-- An ellipse with equation x^2 + ky^2 = 1, where k is a positive real number -/
structure Ellipse (k : ℝ) : Type :=
  (eq : ∀ x y : ℝ, x^2 + k * y^2 = 1)

/-- The focus of the ellipse is on the y-axis -/
def focus_on_y_axis (e : Ellipse k) : Prop :=
  k < 1

/-- The length of the major axis is twice that of the minor axis -/
def major_axis_twice_minor (e : Ellipse k) : Prop :=
  2 * (1 / Real.sqrt k) = 4

/-- Theorem: For an ellipse with the given properties, k = 1/4 -/
theorem ellipse_k_value (k : ℝ) (e : Ellipse k) 
  (h1 : focus_on_y_axis e) (h2 : major_axis_twice_minor e) : k = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_value_l1125_112517


namespace NUMINAMATH_CALUDE_negation_existential_quadratic_l1125_112506

theorem negation_existential_quadratic (x : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_existential_quadratic_l1125_112506


namespace NUMINAMATH_CALUDE_point_on_line_l1125_112587

/-- Given a line with equation x + 2y + 5 = 0, if (m, n) and (m + 2, n + k) are two points on this line,
    then k = -1 -/
theorem point_on_line (m n k : ℝ) : 
  (m + 2*n + 5 = 0) → ((m + 2) + 2*(n + k) + 5 = 0) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1125_112587


namespace NUMINAMATH_CALUDE_sqrt_16_minus_2_squared_equals_zero_l1125_112569

theorem sqrt_16_minus_2_squared_equals_zero : 
  Real.sqrt 16 - 2^2 = 0 := by sorry

end NUMINAMATH_CALUDE_sqrt_16_minus_2_squared_equals_zero_l1125_112569


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l1125_112554

/-- The jumping distances of animals in a contest -/
structure JumpingContest where
  frog_jump : ℕ
  frog_grasshopper_diff : ℕ
  frog_mouse_diff : ℕ

/-- Theorem stating the grasshopper's jump distance given the conditions -/
theorem grasshopper_jump_distance (contest : JumpingContest) 
  (h1 : contest.frog_jump = 58)
  (h2 : contest.frog_grasshopper_diff = 39) :
  contest.frog_jump - contest.frog_grasshopper_diff = 19 := by
  sorry

#check grasshopper_jump_distance

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l1125_112554


namespace NUMINAMATH_CALUDE_fraction_equality_l1125_112542

theorem fraction_equality : (8 : ℚ) / (5 * 48) = 0.8 / (5 * 0.48) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1125_112542


namespace NUMINAMATH_CALUDE_wall_width_calculation_l1125_112527

/-- Given a wall with specific proportions and volume, calculate its width -/
theorem wall_width_calculation (w h l : ℝ) (V : ℝ) (h_def : h = 6 * w) (l_def : l = 7 * h^2) (V_def : V = w * h * l) (V_val : V = 86436) :
  w = (86436 / 1512) ^ (1/4) := by
sorry

end NUMINAMATH_CALUDE_wall_width_calculation_l1125_112527


namespace NUMINAMATH_CALUDE_sector_central_angle_l1125_112589

theorem sector_central_angle (perimeter area : ℝ) (h1 : perimeter = 10) (h2 : area = 4) :
  ∃ (r l θ : ℝ), r > 0 ∧ l > 0 ∧ θ > 0 ∧ 
  2 * r + l = perimeter ∧ 
  1/2 * r * l = area ∧ 
  θ = l / r ∧ 
  θ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1125_112589
