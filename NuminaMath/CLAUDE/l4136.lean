import Mathlib

namespace NUMINAMATH_CALUDE_golden_comets_ratio_l4136_413663

/-- Represents the number of chickens in a flock -/
structure ChickenFlock where
  rhodeIslandReds : ℕ
  goldenComets : ℕ

/-- Given information about Susie's and Britney's chicken flocks -/
def susie : ChickenFlock := { rhodeIslandReds := 11, goldenComets := 6 }
def britney : ChickenFlock :=
  { rhodeIslandReds := 2 * susie.rhodeIslandReds,
    goldenComets := susie.rhodeIslandReds + susie.goldenComets + 8 - (2 * susie.rhodeIslandReds) }

/-- The theorem to be proved -/
theorem golden_comets_ratio :
  2 * britney.goldenComets = susie.goldenComets := by sorry

end NUMINAMATH_CALUDE_golden_comets_ratio_l4136_413663


namespace NUMINAMATH_CALUDE_intersection_height_l4136_413676

/-- Triangle ABC with vertices A(0, 7), B(3, 0), and C(9, 0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Horizontal line y = t intersecting AB at T and AC at U -/
structure Intersection (ABC : Triangle) (t : ℝ) :=
  (T : ℝ × ℝ)
  (U : ℝ × ℝ)

/-- The area of a triangle given three points -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem intersection_height (ABC : Triangle) (t : ℝ) (intr : Intersection ABC t) :
  ABC.A = (0, 7) ∧ ABC.B = (3, 0) ∧ ABC.C = (9, 0) →
  triangleArea ABC.A intr.T intr.U = 18 →
  t = 7 - Real.sqrt 42 :=
by sorry

end NUMINAMATH_CALUDE_intersection_height_l4136_413676


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l4136_413671

theorem least_k_for_inequality : ∃ k : ℤ, k = 5 ∧ 
  (∀ n : ℤ, 0.0010101 * (10 : ℝ)^n > 10 → n ≥ k) ∧
  (0.0010101 * (10 : ℝ)^k > 10) := by
  sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l4136_413671


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l4136_413649

theorem complex_modulus_problem (z : ℂ) : z * (1 + Complex.I) = 1 - Complex.I → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l4136_413649


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_rope_l4136_413632

/-- Represents the lengths of the sides of an isosceles triangle. -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ

/-- 
Given a rope of length 24 cm used to form an isosceles triangle with a base of 6 cm,
prove that the lengths of the three sides of the triangle are 6 cm, 9 cm, and 9 cm.
-/
theorem isosceles_triangle_from_rope (rope_length : ℝ) (triangle : IsoscelesTriangle) :
  rope_length = 24 →
  triangle.base = 6 →
  triangle.base + 2 * triangle.side = rope_length →
  triangle.base = 6 ∧ triangle.side = 9 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_from_rope_l4136_413632


namespace NUMINAMATH_CALUDE_triangle_function_k_range_l4136_413687

-- Define the function f(x) = kx + 2
def f (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the property of being a "triangle function" on a domain
def is_triangle_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ (x y z : ℝ), a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ a ≤ z ∧ z ≤ b →
    f x + f y > f z ∧ f y + f z > f x ∧ f z + f x > f y

-- State the theorem
theorem triangle_function_k_range :
  ∀ k : ℝ, is_triangle_function (f k) 1 4 ↔ -2/7 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_function_k_range_l4136_413687


namespace NUMINAMATH_CALUDE_complex_square_root_l4136_413635

theorem complex_square_root (z : ℂ) : z^2 = -4 ∧ z.im > 0 → z = 2*I :=
sorry

end NUMINAMATH_CALUDE_complex_square_root_l4136_413635


namespace NUMINAMATH_CALUDE_median_and_mode_are_50_l4136_413636

/-- Represents a speed measurement and its frequency --/
structure SpeedData where
  speed : ℕ
  frequency : ℕ

/-- The dataset of vehicle speeds and their frequencies --/
def speedDataset : List SpeedData := [
  ⟨48, 5⟩,
  ⟨49, 4⟩,
  ⟨50, 8⟩,
  ⟨51, 2⟩,
  ⟨52, 1⟩
]

/-- Calculates the median of the dataset --/
def calculateMedian (data : List SpeedData) : ℕ := sorry

/-- Calculates the mode of the dataset --/
def calculateMode (data : List SpeedData) : ℕ := sorry

/-- Theorem stating that the median and mode of the dataset are both 50 --/
theorem median_and_mode_are_50 :
  calculateMedian speedDataset = 50 ∧ calculateMode speedDataset = 50 := by sorry

end NUMINAMATH_CALUDE_median_and_mode_are_50_l4136_413636


namespace NUMINAMATH_CALUDE_midpoint_coordinates_l4136_413637

/-- Given point A and vector AB, prove that the midpoint of segment AB has specific coordinates -/
theorem midpoint_coordinates (A B : ℝ × ℝ) (h1 : A = (-3, 2)) (h2 : B - A = (6, 0)) :
  (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 2 := by
  sorry

#check midpoint_coordinates

end NUMINAMATH_CALUDE_midpoint_coordinates_l4136_413637


namespace NUMINAMATH_CALUDE_place_value_ratio_l4136_413666

-- Define the number
def number : ℚ := 53674.9281

-- Define the place value of a digit at a specific position
def place_value (n : ℚ) (pos : ℤ) : ℚ := 10 ^ pos

-- Define the position of digit 6 (counting from right, with decimal point at 0)
def pos_6 : ℤ := 3

-- Define the position of digit 8 (counting from right, with decimal point at 0)
def pos_8 : ℤ := -1

-- Theorem to prove
theorem place_value_ratio :
  (place_value number pos_6) / (place_value number pos_8) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratio_l4136_413666


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4136_413628

theorem sqrt_equation_solution :
  let f (x : ℝ) := Real.sqrt (3 * x - 5) + 14 / Real.sqrt (3 * x - 5)
  ∀ x : ℝ, f x = 8 ↔ x = (23 + 8 * Real.sqrt 2) / 3 ∨ x = (23 - 8 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4136_413628


namespace NUMINAMATH_CALUDE_dan_placed_16_pencils_l4136_413609

/-- The number of pencils Dan placed on the desk -/
def pencils_dan_placed (drawer : ℕ) (desk_initial : ℕ) (total_after : ℕ) : ℕ :=
  total_after - (drawer + desk_initial)

/-- Theorem stating that Dan placed 16 pencils on the desk -/
theorem dan_placed_16_pencils : 
  pencils_dan_placed 43 19 78 = 16 := by
  sorry

end NUMINAMATH_CALUDE_dan_placed_16_pencils_l4136_413609


namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l4136_413655

/-- The maximum number of students for equal distribution of pens and pencils -/
theorem max_students_equal_distribution (pens pencils : ℕ) : 
  pens = 891 → pencils = 810 → 
  (∃ (max_students : ℕ), 
    max_students = Nat.gcd pens pencils ∧ 
    max_students > 0 ∧
    pens % max_students = 0 ∧ 
    pencils % max_students = 0 ∧
    ∀ (n : ℕ), n > max_students → (pens % n ≠ 0 ∨ pencils % n ≠ 0)) := by
  sorry

#eval Nat.gcd 891 810  -- Expected output: 81

end NUMINAMATH_CALUDE_max_students_equal_distribution_l4136_413655


namespace NUMINAMATH_CALUDE_five_thursdays_in_august_l4136_413691

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Counts the occurrences of a specific day in a month -/
def countDayInMonth (startDay : DayOfWeek) (daysInMonth : Nat) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Theorem: If July has five Tuesdays and both July and August have 31 days, 
    then Thursday must occur five times in August of the same year -/
theorem five_thursdays_in_august 
  (july_start : DayOfWeek) 
  (h1 : countDayInMonth july_start 31 DayOfWeek.Tuesday = 5) 
  : ∃ (august_start : DayOfWeek), 
    countDayInMonth august_start 31 DayOfWeek.Thursday = 5 :=
  sorry

end NUMINAMATH_CALUDE_five_thursdays_in_august_l4136_413691


namespace NUMINAMATH_CALUDE_no_roots_for_equation_l4136_413653

theorem no_roots_for_equation : ∀ x : ℝ, ¬(Real.sqrt (7 - x) = x * Real.sqrt (7 - x) - 1) := by
  sorry

end NUMINAMATH_CALUDE_no_roots_for_equation_l4136_413653


namespace NUMINAMATH_CALUDE_seven_students_distribution_l4136_413616

/-- The number of ways to distribute n students into two dormitories with at least m students in each -/
def distribution_plans (n : ℕ) (m : ℕ) : ℕ :=
  (Finset.sum (Finset.range (n - 2*m + 1)) (λ k => (Nat.choose n (m + k)) * 2)) * 2

/-- Theorem: There are 112 ways to distribute 7 students into two dormitories with at least 2 students in each -/
theorem seven_students_distribution : distribution_plans 7 2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_seven_students_distribution_l4136_413616


namespace NUMINAMATH_CALUDE_fourth_to_second_quadrant_l4136_413608

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Predicate to check if a point is in the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem stating that if P(a,b) is in the fourth quadrant, then Q(-a,-b) is in the second quadrant -/
theorem fourth_to_second_quadrant (p : Point) :
  is_in_fourth_quadrant p → is_in_second_quadrant (Point.mk (-p.x) (-p.y)) := by
  sorry

end NUMINAMATH_CALUDE_fourth_to_second_quadrant_l4136_413608


namespace NUMINAMATH_CALUDE_min_operations_to_measure_88_l4136_413662

/-- Represents the state of the puzzle -/
structure PuzzleState where
  barrel : ℕ
  vessel7 : ℕ
  vessel5 : ℕ

/-- Represents a pouring operation -/
inductive PourOperation
  | FillFrom7 : PourOperation
  | FillFrom5 : PourOperation
  | EmptyTo7 : PourOperation
  | EmptyTo5 : PourOperation
  | Pour7To5 : PourOperation
  | Pour5To7 : PourOperation

/-- Applies a single pouring operation to a puzzle state -/
def applyOperation (state : PuzzleState) (op : PourOperation) : PuzzleState :=
  sorry

/-- Checks if a sequence of operations is valid and results in the target state -/
def isValidSequence (initialState : PuzzleState) (targetBarrel : ℕ) (ops : List PourOperation) : Bool :=
  sorry

/-- Theorem: The minimum number of operations to measure 88 quarts is 17 -/
theorem min_operations_to_measure_88 :
  ∃ (ops : List PourOperation),
    ops.length = 17 ∧
    isValidSequence (PuzzleState.mk 108 0 0) 88 ops ∧
    ∀ (other_ops : List PourOperation),
      isValidSequence (PuzzleState.mk 108 0 0) 88 other_ops →
      other_ops.length ≥ 17 :=
  sorry

end NUMINAMATH_CALUDE_min_operations_to_measure_88_l4136_413662


namespace NUMINAMATH_CALUDE_power_product_sum_equality_l4136_413683

theorem power_product_sum_equality : (3^5 * 6^3) + 3^3 = 52515 := by
  sorry

end NUMINAMATH_CALUDE_power_product_sum_equality_l4136_413683


namespace NUMINAMATH_CALUDE_athletes_arrangement_l4136_413643

/-- The number of ways to arrange athletes from three teams in a row -/
def arrange_athletes (team_a : ℕ) (team_b : ℕ) (team_c : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial team_a) * (Nat.factorial team_b) * (Nat.factorial team_c)

/-- Theorem: The number of ways to arrange 10 athletes from 3 teams (with 4, 3, and 3 athletes respectively) in a row, where athletes from the same team must sit together, is 5184 -/
theorem athletes_arrangement :
  arrange_athletes 4 3 3 = 5184 :=
by sorry

end NUMINAMATH_CALUDE_athletes_arrangement_l4136_413643


namespace NUMINAMATH_CALUDE_dice_configuration_dots_l4136_413646

/-- Represents a die face with a number of dots -/
structure DieFace where
  dots : Nat
  valid : dots ≥ 1 ∧ dots ≤ 6

/-- Represents a die with six faces -/
structure Die where
  faces : Fin 6 → DieFace
  sum_opposite : ∀ i : Fin 3, (faces i).dots + (faces (i + 3)).dots = 7

/-- Represents the configuration of 4 dice glued together -/
structure DiceConfiguration where
  dice : Fin 4 → Die
  face_c : DieFace
  face_c_is_six : face_c.dots = 6

/-- The theorem to be proved -/
theorem dice_configuration_dots (config : DiceConfiguration) :
  ∃ (face_a face_b face_d : DieFace),
    face_a.dots = 3 ∧
    face_b.dots = 5 ∧
    config.face_c.dots = 6 ∧
    face_d.dots = 5 := by
  sorry

end NUMINAMATH_CALUDE_dice_configuration_dots_l4136_413646


namespace NUMINAMATH_CALUDE_expand_product_l4136_413610

theorem expand_product (x : ℝ) : (x + 3) * (x + 6) = x^2 + 9*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l4136_413610


namespace NUMINAMATH_CALUDE_ratio_calculations_l4136_413647

theorem ratio_calculations (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 6) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 ∧ 
  (A + C) / (2 * B + A) = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculations_l4136_413647


namespace NUMINAMATH_CALUDE_cheese_fries_cost_is_eight_l4136_413612

/-- Represents the cost of items and money brought by Jim and his cousin --/
structure RestaurantScenario where
  cheeseburger_cost : ℚ
  milkshake_cost : ℚ
  jim_money : ℚ
  cousin_money : ℚ
  spent_percentage : ℚ

/-- Calculates the cost of cheese fries given a RestaurantScenario --/
def cheese_fries_cost (scenario : RestaurantScenario) : ℚ :=
  let total_money := scenario.jim_money + scenario.cousin_money
  let total_spent := scenario.spent_percentage * total_money
  let burger_shake_cost := 2 * (scenario.cheeseburger_cost + scenario.milkshake_cost)
  total_spent - burger_shake_cost

/-- Theorem stating that the cost of cheese fries is 8 given the specific scenario --/
theorem cheese_fries_cost_is_eight :
  let scenario := {
    cheeseburger_cost := 3,
    milkshake_cost := 5,
    jim_money := 20,
    cousin_money := 10,
    spent_percentage := 4/5
  }
  cheese_fries_cost scenario = 8 := by
  sorry


end NUMINAMATH_CALUDE_cheese_fries_cost_is_eight_l4136_413612


namespace NUMINAMATH_CALUDE_sum_of_r_values_l4136_413602

/-- Given two quadratic equations with a common real root, prove the sum of possible values of r -/
theorem sum_of_r_values (r : ℝ) : 
  (∃ x : ℝ, x^2 + (r-1)*x + 6 = 0 ∧ x^2 + (2*r+1)*x + 22 = 0) → 
  (∃ r1 r2 : ℝ, (r = r1 ∨ r = r2) ∧ r1 + r2 = 12/5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_r_values_l4136_413602


namespace NUMINAMATH_CALUDE_price_increase_percentage_l4136_413688

theorem price_increase_percentage (x : ℝ) : 
  (∀ (P : ℝ), P > 0 → P * (1 + x / 100) * (1 - 20 / 100) = P) → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l4136_413688


namespace NUMINAMATH_CALUDE_power_function_increasing_l4136_413679

theorem power_function_increasing (m : ℝ) : 
  (∀ x > 0, ∀ h > 0, (m^2 - 4*m + 1) * x^(m^2 - 2*m - 3) < (m^2 - 4*m + 1) * (x + h)^(m^2 - 2*m - 3)) ↔ 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_power_function_increasing_l4136_413679


namespace NUMINAMATH_CALUDE_shirts_count_l4136_413614

/-- Given a ratio of pants : shorts : shirts and the number of pants, 
    calculate the number of shirts -/
def calculate_shirts (pants_ratio : ℕ) (shorts_ratio : ℕ) (shirts_ratio : ℕ) 
                     (num_pants : ℕ) : ℕ :=
  (num_pants / pants_ratio) * shirts_ratio

/-- Prove that given the ratio 7 : 7 : 10 for pants : shorts : shirts, 
    and 14 pants, there are 20 shirts -/
theorem shirts_count : calculate_shirts 7 7 10 14 = 20 := by
  sorry

end NUMINAMATH_CALUDE_shirts_count_l4136_413614


namespace NUMINAMATH_CALUDE_distance_to_asymptote_l4136_413664

-- Define the parabola C₁
def C₁ (a : ℝ) (x y : ℝ) : Prop := y^2 = 8*a*x ∧ a > 0

-- Define the line l
def l (a : ℝ) (x y : ℝ) : Prop := y = x - 2*a

-- Define the hyperbola C₂
def C₂ (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the directrix of C₁
def directrix (a : ℝ) : ℝ := -2*a

-- Define the focus of C₁
def focus_C₁ (a : ℝ) : ℝ × ℝ := (2*a, 0)

-- Define the asymptote of C₂
def asymptote_C₂ (a b : ℝ) (x y : ℝ) : Prop := b*x - a*y = 0

-- Main theorem
theorem distance_to_asymptote 
  (a b : ℝ) 
  (h₁ : C₁ a (2*a) 0)  -- C₁ passes through its focus
  (h₂ : ∃ x y, C₁ a x y ∧ l a x y ∧ (x - 2*a)^2 + y^2 = 256)  -- Segment length is 16
  (h₃ : ∃ x, C₂ a b x (directrix a))  -- One focus of C₂ on directrix of C₁
  : (abs (2*a)) / Real.sqrt (b^2 + a^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_distance_to_asymptote_l4136_413664


namespace NUMINAMATH_CALUDE_three_digit_number_rearrangement_l4136_413684

theorem three_digit_number_rearrangement (a b c : ℕ) : 
  a < 10 → b < 10 → c < 10 → a ≠ 0 →
  (100 * a + 10 * b + c) + 
  (100 * a + 10 * c + b) + 
  (100 * b + 10 * c + a) + 
  (100 * b + 10 * a + c) + 
  (100 * c + 10 * a + b) + 
  (100 * c + 10 * b + a) = 4422 →
  a + b + c ≥ 18 →
  100 * a + 10 * b + c = 785 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_rearrangement_l4136_413684


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l4136_413660

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l4136_413660


namespace NUMINAMATH_CALUDE_syllogism_conclusion_l4136_413673

-- Define the sets
variable (U : Type) -- Universe set
variable (Mem : Set U) -- Set of Mems
variable (En : Set U) -- Set of Ens
variable (Veen : Set U) -- Set of Veens

-- Define the hypotheses
variable (h1 : Mem ⊆ En) -- All Mems are Ens
variable (h2 : ∃ x, x ∈ En ∩ Veen) -- Some Ens are Veens

-- Define the conclusions to be proved
def conclusion1 : Prop := ∃ x, x ∈ Mem ∩ Veen -- Some Mems are Veens
def conclusion2 : Prop := ∃ x, x ∈ Veen \ Mem -- Some Veens are not Mems

-- Theorem statement
theorem syllogism_conclusion (U : Type) (Mem En Veen : Set U) 
  (h1 : Mem ⊆ En) (h2 : ∃ x, x ∈ En ∩ Veen) : 
  conclusion1 U Mem Veen ∧ conclusion2 U Mem Veen := by
  sorry

end NUMINAMATH_CALUDE_syllogism_conclusion_l4136_413673


namespace NUMINAMATH_CALUDE_cylinder_volume_l4136_413685

/-- Represents a cylinder formed by rotating a rectangle around one of its sides. -/
structure Cylinder where
  /-- The area of the original rectangle. -/
  S : ℝ
  /-- The circumference of the circle described by the intersection point of the rectangle's diagonals. -/
  C : ℝ
  /-- Ensure that S and C are positive. -/
  S_pos : S > 0
  C_pos : C > 0

/-- The volume of the cylinder. -/
def volume (cyl : Cylinder) : ℝ := cyl.S * cyl.C

/-- Theorem stating that the volume of the cylinder is equal to the product of S and C. -/
theorem cylinder_volume (cyl : Cylinder) : volume cyl = cyl.S * cyl.C := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l4136_413685


namespace NUMINAMATH_CALUDE_geometric_sequence_product_equality_l4136_413686

/-- Given four non-zero real numbers, prove that forming a geometric sequence
    is sufficient but not necessary for their product equality. -/
theorem geometric_sequence_product_equality (a b c d : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) → a * d = b * c ∧
  ∃ a' b' c' d' : ℝ, a' * d' = b' * c' ∧ ¬(∃ r : ℝ, b' = a' * r ∧ c' = b' * r ∧ d' = c' * r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_equality_l4136_413686


namespace NUMINAMATH_CALUDE_staff_distribution_theorem_l4136_413672

def distribute_staff (n : ℕ) (k : ℕ) : ℕ :=
  let arrangements := (n.choose 1 * (n-1).choose 1) / 2 +
                      (n.choose 2 * (n-2).choose 2) / 2 +
                      (n.choose 3 * (n-3).choose 3) / 2
  arrangements * (k.factorial)

theorem staff_distribution_theorem :
  distribute_staff 7 3 = 1176 := by
  sorry

end NUMINAMATH_CALUDE_staff_distribution_theorem_l4136_413672


namespace NUMINAMATH_CALUDE_missing_village_population_l4136_413670

theorem missing_village_population
  (total_villages : ℕ)
  (average_population : ℕ)
  (known_populations : List ℕ)
  (h1 : total_villages = 7)
  (h2 : average_population = 1000)
  (h3 : known_populations = [900, 1100, 1023, 945, 980, 1249])
  (h4 : known_populations.length = total_villages - 1) :
  total_villages * average_population - known_populations.sum = 803 := by
  sorry

end NUMINAMATH_CALUDE_missing_village_population_l4136_413670


namespace NUMINAMATH_CALUDE_cafe_customers_l4136_413678

/-- Prove that the number of customers in a group is 12, given the following conditions:
  * 3 offices ordered 10 sandwiches each
  * Half of the group ordered 4 sandwiches each
  * Total sandwiches made is 54
-/
theorem cafe_customers (num_offices : Nat) (sandwiches_per_office : Nat)
  (sandwiches_per_customer : Nat) (total_sandwiches : Nat) :
  num_offices = 3 →
  sandwiches_per_office = 10 →
  sandwiches_per_customer = 4 →
  total_sandwiches = 54 →
  ∃ (num_customers : Nat),
    num_customers = 12 ∧
    total_sandwiches = num_offices * sandwiches_per_office +
      (num_customers / 2) * sandwiches_per_customer :=
by
  sorry

end NUMINAMATH_CALUDE_cafe_customers_l4136_413678


namespace NUMINAMATH_CALUDE_total_trip_time_l4136_413682

/-- Given that Tim drove for 5 hours and was stuck in traffic for twice as long as he was driving,
    prove that the total trip time is 15 hours. -/
theorem total_trip_time (driving_time : ℕ) (traffic_time : ℕ) : 
  driving_time = 5 →
  traffic_time = 2 * driving_time →
  driving_time + traffic_time = 15 :=
by sorry

end NUMINAMATH_CALUDE_total_trip_time_l4136_413682


namespace NUMINAMATH_CALUDE_square_perimeter_side_ratio_l4136_413621

theorem square_perimeter_side_ratio (s : ℝ) (hs : s > 0) :
  let new_side := s + 1
  let new_perimeter := 4 * new_side
  new_perimeter / new_side = 4 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_side_ratio_l4136_413621


namespace NUMINAMATH_CALUDE_ticket_price_possibilities_l4136_413693

theorem ticket_price_possibilities (x : ℕ) : 
  (∃ n m : ℕ, n * x = 72 ∧ m * x = 108 ∧ Even x) ↔ 
  x ∈ ({2, 4, 6, 12, 18, 36} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_ticket_price_possibilities_l4136_413693


namespace NUMINAMATH_CALUDE_passes_through_origin_parallel_to_line_intersects_below_l4136_413630

/-- Linear function definition -/
def linear_function (m x : ℝ) : ℝ := (2*m + 1)*x + m - 3

/-- Theorem for when the function passes through the origin -/
theorem passes_through_origin (m : ℝ) : 
  linear_function m 0 = 0 ↔ m = 3 := by sorry

/-- Theorem for when the function is parallel to y = 3x - 3 -/
theorem parallel_to_line (m : ℝ) :
  (2*m + 1 = 3) ↔ m = 1 := by sorry

/-- Theorem for when the function intersects y-axis below x-axis -/
theorem intersects_below (m : ℝ) :
  (linear_function m 0 < 0 ∧ 2*m + 1 ≠ 0) ↔ (m < 3 ∧ m ≠ -1/2) := by sorry

end NUMINAMATH_CALUDE_passes_through_origin_parallel_to_line_intersects_below_l4136_413630


namespace NUMINAMATH_CALUDE_arithmetic_equation_l4136_413603

theorem arithmetic_equation : 12 - 11 + (9 * 8) + 7 - (6 * 5) + 4 - 3 = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l4136_413603


namespace NUMINAMATH_CALUDE_quadratic_minimum_l4136_413631

theorem quadratic_minimum (x : ℝ) : 4 * x^2 + 8 * x + 16 ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l4136_413631


namespace NUMINAMATH_CALUDE_option2_expected_cost_l4136_413618

/-- Represents the water temperature situations -/
inductive WaterTemp
  | Normal
  | SlightlyHigh
  | ExtremelyHigh

/-- Probability of extremely high water temperature -/
def probExtremelyHigh : ℝ := 0.01

/-- Probability of slightly high water temperature -/
def probSlightlyHigh : ℝ := 0.25

/-- Loss incurred when water temperature is extremely high -/
def lossExtremelyHigh : ℝ := 600000

/-- Loss incurred when water temperature is slightly high -/
def lossSlightlyHigh : ℝ := 100000

/-- Cost of implementing Option 2 (temperature control equipment) -/
def costOption2 : ℝ := 20000

/-- Expected cost of Option 2 -/
def expectedCostOption2 : ℝ := 
  (lossExtremelyHigh + costOption2) * probExtremelyHigh + costOption2 * (1 - probExtremelyHigh)

/-- Theorem stating that the expected cost of Option 2 is 2600 yuan -/
theorem option2_expected_cost : expectedCostOption2 = 2600 := by
  sorry

end NUMINAMATH_CALUDE_option2_expected_cost_l4136_413618


namespace NUMINAMATH_CALUDE_pentagon_side_length_l4136_413681

/-- A five-sided figure with equal side lengths -/
structure Pentagon where
  side_length : ℝ
  perimeter : ℝ
  side_count : ℕ := 5
  all_sides_equal : perimeter = side_count * side_length

/-- Theorem: Given a pentagon with perimeter 23.4 cm, the length of one side is 4.68 cm -/
theorem pentagon_side_length (p : Pentagon) (h : p.perimeter = 23.4) : p.side_length = 4.68 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_side_length_l4136_413681


namespace NUMINAMATH_CALUDE_symmetric_points_l4136_413642

/-- Given a line ax + by + c = 0 and two points (x₁, y₁) and (x₂, y₂), 
    this function returns true if the points are symmetric with respect to the line -/
def are_symmetric (a b c : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  -- The midpoint of the two points lies on the line
  a * ((x₁ + x₂) / 2) + b * ((y₁ + y₂) / 2) + c = 0 ∧
  -- The line connecting the two points is perpendicular to the given line
  (y₂ - y₁) * a = -(x₂ - x₁) * b

/-- Theorem stating that (-5, -4) is symmetric to (3, 4) with respect to the line x + y + 1 = 0 -/
theorem symmetric_points : are_symmetric 1 1 1 3 4 (-5) (-4) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_l4136_413642


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l4136_413623

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  n ≥ 3 → 
  interior_angle = 108 → 
  (n : ℝ) * (180 - interior_angle) = 360 → 
  n = 5 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l4136_413623


namespace NUMINAMATH_CALUDE_disjoint_circles_condition_l4136_413654

def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4
def circle2 (x y a : ℝ) : Prop := x^2 + (y - a)^2 = 1

def circles_disjoint (a : ℝ) : Prop :=
  ∀ x y, ¬(circle1 x y ∧ circle2 x y a)

theorem disjoint_circles_condition (a : ℝ) :
  circles_disjoint a ↔ (a > 1 + 2 * Real.sqrt 2 ∨ a < 1 - 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_disjoint_circles_condition_l4136_413654


namespace NUMINAMATH_CALUDE_annual_sales_profit_scientific_notation_l4136_413667

/-- Represents the annual sales profit in yuan -/
def annual_sales_profit : ℝ := 1.5e12

/-- Expresses the annual sales profit in scientific notation -/
def scientific_notation : ℝ := 1.5 * (10 ^ 12)

theorem annual_sales_profit_scientific_notation : 
  annual_sales_profit = scientific_notation := by sorry

end NUMINAMATH_CALUDE_annual_sales_profit_scientific_notation_l4136_413667


namespace NUMINAMATH_CALUDE_passing_percentage_l4136_413648

def total_marks : ℕ := 400
def student_marks : ℕ := 100
def failing_margin : ℕ := 40

theorem passing_percentage :
  (student_marks + failing_margin) * 100 / total_marks = 35 := by
sorry

end NUMINAMATH_CALUDE_passing_percentage_l4136_413648


namespace NUMINAMATH_CALUDE_rotation_of_specific_point_l4136_413669

/-- Rotation of a 2D vector clockwise by π/2 radians -/
def rotate_clockwise_pi_over_2 (x y : ℝ) : ℝ × ℝ :=
  (y, -x)

theorem rotation_of_specific_point :
  let A : ℝ × ℝ := (-1/2, Real.sqrt 3/2)
  let OA' := rotate_clockwise_pi_over_2 A.1 A.2
  OA' = (Real.sqrt 3/2, 1/2) := by
  sorry

end NUMINAMATH_CALUDE_rotation_of_specific_point_l4136_413669


namespace NUMINAMATH_CALUDE_solution_set_inequality_l4136_413615

theorem solution_set_inequality (x : ℝ) :
  (abs x - 2) * (x - 1) ≥ 0 ↔ -2 ≤ x ∧ x ≤ 1 ∨ x ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l4136_413615


namespace NUMINAMATH_CALUDE_large_kangaroo_count_toy_store_kangaroos_l4136_413624

theorem large_kangaroo_count (total : ℕ) (empty_pouch : ℕ) (small_per_pouch : ℕ) : ℕ :=
  let full_pouch := total - empty_pouch
  let small_kangaroos := full_pouch * small_per_pouch
  total - small_kangaroos

theorem toy_store_kangaroos :
  large_kangaroo_count 100 77 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_large_kangaroo_count_toy_store_kangaroos_l4136_413624


namespace NUMINAMATH_CALUDE_sum_of_digits_theorem_l4136_413601

-- Define the set A
def A : Set Int := {m | ∃ p q : Int, p > 0 ∧ q > 0 ∧ p * q = 2020 ∧ p + q = -m}

-- Define the set B
def B : Set Int := {n | ∃ r s : Int, r > 0 ∧ s > 0 ∧ r * s = n ∧ r + s = 2020}

-- Define a function to calculate the sum of digits
def sumOfDigits (n : Int) : Nat :=
  (n.natAbs.digits 10).sum

-- State the theorem
theorem sum_of_digits_theorem :
  ∃ a b : Int, a ∈ A ∧ b ∈ B ∧ (∀ m ∈ A, m ≤ a) ∧ (∀ n ∈ B, b ≤ n) ∧
  sumOfDigits (a + b) = 27 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_theorem_l4136_413601


namespace NUMINAMATH_CALUDE_sum_digits_count_numeric_hex_below_2000_l4136_413692

/-- Converts a decimal number to hexadecimal --/
def decimalToHex (n : ℕ) : String := sorry

/-- Counts positive hexadecimal numbers below a given hexadecimal number
    that contain only numeric digits (0-9) --/
def countNumericHex (hex : String) : ℕ := sorry

/-- Sums the digits of a natural number --/
def sumDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of the digits of the count of positive hexadecimal numbers
    below the hexadecimal representation of 2000 that contain only numeric digits (0-9) is 25 --/
theorem sum_digits_count_numeric_hex_below_2000 :
  sumDigits (countNumericHex (decimalToHex 2000)) = 25 := by sorry

end NUMINAMATH_CALUDE_sum_digits_count_numeric_hex_below_2000_l4136_413692


namespace NUMINAMATH_CALUDE_inverse_g_at_124_l4136_413611

noncomputable def g (x : ℝ) : ℝ := 5 * x^3 - 4 * x + 1

theorem inverse_g_at_124 : g⁻¹ 124 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_at_124_l4136_413611


namespace NUMINAMATH_CALUDE_problem_statement_l4136_413677

theorem problem_statement : (5/12 : ℝ)^2022 * (-2.4)^2023 = -12/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4136_413677


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l4136_413644

theorem triangle_abc_properties (a b : ℝ) (cosB : ℝ) (S : ℝ) :
  a = 5 →
  b = 6 →
  cosB = -4/5 →
  S = 15 * Real.sqrt 7 / 4 →
  ∃ (A R c : ℝ),
    (A = π/6 ∧ R = 5) ∧
    (c = 4 ∨ c = Real.sqrt 106) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l4136_413644


namespace NUMINAMATH_CALUDE_haydens_tank_water_remaining_l4136_413606

/-- Calculates the amount of water remaining in a tank after a given time period,
    considering initial volume, loss rate, and water additions. -/
def water_remaining (initial_volume : ℝ) (loss_rate : ℝ) (time : ℕ) (additions : List ℝ) : ℝ :=
  initial_volume - loss_rate * time + additions.sum

/-- Theorem stating that given the specific conditions of Hayden's tank,
    the amount of water remaining after 4 hours is 36 gallons. -/
theorem haydens_tank_water_remaining :
  let initial_volume : ℝ := 40
  let loss_rate : ℝ := 2
  let time : ℕ := 4
  let additions : List ℝ := [0, 0, 1, 3]
  water_remaining initial_volume loss_rate time additions = 36 := by
  sorry

end NUMINAMATH_CALUDE_haydens_tank_water_remaining_l4136_413606


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4136_413697

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 < 0)) ↔ (∃ x : ℝ, x^3 - x^2 + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4136_413697


namespace NUMINAMATH_CALUDE_total_food_consumed_theorem_l4136_413629

/-- Represents the amount of food a dog eats per meal -/
structure MealPortion where
  dry : Float
  wet : Float

/-- Represents the feeding schedule for a dog -/
structure FeedingSchedule where
  portion : MealPortion
  mealsPerDay : Nat

/-- Conversion rates for dry and wet food -/
def dryFoodConversion : Float := 3.2  -- cups per pound
def wetFoodConversion : Float := 2.8  -- cups per pound

/-- Feeding schedules for each dog -/
def momoSchedule : FeedingSchedule := { portion := { dry := 1.3, wet := 0.7 }, mealsPerDay := 2 }
def fifiSchedule : FeedingSchedule := { portion := { dry := 1.6, wet := 0.5 }, mealsPerDay := 2 }
def gigiSchedule : FeedingSchedule := { portion := { dry := 2.0, wet := 1.0 }, mealsPerDay := 3 }

/-- Calculate total food consumed by all dogs in pounds -/
def totalFoodConsumed (momo fifi gigi : FeedingSchedule) : Float :=
  let totalDry := (momo.portion.dry * momo.mealsPerDay.toFloat +
                   fifi.portion.dry * fifi.mealsPerDay.toFloat +
                   gigi.portion.dry * gigi.mealsPerDay.toFloat) / dryFoodConversion
  let totalWet := (momo.portion.wet * momo.mealsPerDay.toFloat +
                   fifi.portion.wet * fifi.mealsPerDay.toFloat +
                   gigi.portion.wet * gigi.mealsPerDay.toFloat) / wetFoodConversion
  totalDry + totalWet

/-- Theorem: The total amount of food consumed by all three dogs in a day is approximately 5.6161 pounds -/
theorem total_food_consumed_theorem :
  Float.abs (totalFoodConsumed momoSchedule fifiSchedule gigiSchedule - 5.6161) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_total_food_consumed_theorem_l4136_413629


namespace NUMINAMATH_CALUDE_triangle_side_range_l4136_413680

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def validTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = 180

-- Theorem statement
theorem triangle_side_range (x : ℝ) :
  (∃ t1 t2 : Triangle, 
    validTriangle t1 ∧ validTriangle t2 ∧
    t1.b = 2 ∧ t2.b = 2 ∧
    t1.B = 60 ∧ t2.B = 60 ∧
    t1.a = x ∧ t2.a = x ∧
    t1 ≠ t2) →
  2 < x ∧ x < (4 * Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l4136_413680


namespace NUMINAMATH_CALUDE_no_hexagon_cross_section_l4136_413633

/-- Represents the possible shapes that can result from cutting a triangular prism with a plane -/
inductive CrossSectionShape
| Rectangle
| Triangle
| Trapezoid
| Pentagon
| Hexagon

/-- Represents a triangular prism -/
structure TriangularPrism

/-- Represents a plane used for cutting the prism -/
structure CuttingPlane

/-- Represents the result of cutting a triangular prism with a plane -/
def cut (prism : TriangularPrism) (plane : CuttingPlane) : CrossSectionShape := sorry

/-- Theorem stating that cutting a triangular prism with a plane cannot result in a hexagon -/
theorem no_hexagon_cross_section (prism : TriangularPrism) (plane : CuttingPlane) :
  cut prism plane ≠ CrossSectionShape.Hexagon := by
  sorry

end NUMINAMATH_CALUDE_no_hexagon_cross_section_l4136_413633


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4136_413605

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (4 + 3*I) / (1 + 2*I)
  Complex.im z = -1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4136_413605


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l4136_413634

theorem smallest_b_for_factorization : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∃ (r s : ℤ), x^2 + b*x + 1512 = (x + r) * (x + s)) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ¬∃ (r s : ℤ), x^2 + b'*x + 1512 = (x + r) * (x + s)) ∧
  b = 78 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l4136_413634


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l4136_413650

-- Define the repeating decimals
def repeating_decimal_72 : ℚ := 8/11
def repeating_decimal_124 : ℚ := 41/33

-- State the theorem
theorem repeating_decimal_division :
  repeating_decimal_72 / repeating_decimal_124 = 264/451 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l4136_413650


namespace NUMINAMATH_CALUDE_paperboy_delivery_12_l4136_413689

def paperboy_delivery (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | m + 4 => paperboy_delivery m + paperboy_delivery (m + 1) + paperboy_delivery (m + 2) + paperboy_delivery (m + 3)

theorem paperboy_delivery_12 : paperboy_delivery 12 = 2873 := by
  sorry

end NUMINAMATH_CALUDE_paperboy_delivery_12_l4136_413689


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l4136_413696

/-- Represents the dimensions of a rectangular roof --/
structure RoofDimensions where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular roof --/
def area (r : RoofDimensions) : ℝ := r.width * r.length

/-- Theorem: For a rectangular roof with length 4 times its width and an area of 1024 square feet,
    the difference between the length and width is 48 feet. --/
theorem roof_dimension_difference (r : RoofDimensions) 
    (h1 : r.length = 4 * r.width) 
    (h2 : area r = 1024) : 
    r.length - r.width = 48 := by
  sorry


end NUMINAMATH_CALUDE_roof_dimension_difference_l4136_413696


namespace NUMINAMATH_CALUDE_golden_ratio_geometric_sequence_l4136_413641

theorem golden_ratio_geometric_sequence : 
  let x : ℝ := (1 + Real.sqrt 5) / 2
  let int_part := ⌊x⌋
  let frac_part := x - int_part
  (frac_part * x = int_part * int_part) ∧ (int_part * x = x * x) := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_geometric_sequence_l4136_413641


namespace NUMINAMATH_CALUDE_circle_tangent_problem_l4136_413639

/-- Two lines l₁ and l₂ are perpendicular if their slopes multiply to -1 -/
def perpendicular (a : ℝ) : Prop := a * (1/a) = -1

/-- A line ax + by + c = 0 is tangent to the circle x² + y² = r² 
    if the distance from (0,0) to the line equals r -/
def tangent_to_circle (a b c r : ℝ) : Prop :=
  (c / (a^2 + b^2).sqrt)^2 = r^2

theorem circle_tangent_problem (a : ℝ) :
  perpendicular a →
  tangent_to_circle 1 0 2 (b^2).sqrt →
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_problem_l4136_413639


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4136_413607

theorem inverse_variation_problem (x w : ℝ) (k : ℝ) :
  (∀ x w, x^4 * w^(1/4) = k) →
  (3^4 * 16^(1/4) = k) →
  (6^4 * w^(1/4) = k) →
  w = 1 / 4096 :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4136_413607


namespace NUMINAMATH_CALUDE_star_eight_ten_l4136_413638

/-- Custom operation * for rational numbers -/
def star (m n p : ℚ) (x y : ℚ) : ℚ := m * x + n * y + p

/-- Theorem stating that if 3 * 5 = 30 and 4 * 6 = 425, then 8 * 10 = 2005 -/
theorem star_eight_ten (m n p : ℚ) 
  (h1 : star m n p 3 5 = 30)
  (h2 : star m n p 4 6 = 425) : 
  star m n p 8 10 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_star_eight_ten_l4136_413638


namespace NUMINAMATH_CALUDE_suzanna_bike_ride_l4136_413613

/-- Calculates the distance traveled given a constant speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem suzanna_bike_ride :
  let rate : ℝ := 0.75 / 5  -- miles per minute
  let time : ℝ := 45        -- minutes
  distance_traveled rate time = 6.75 := by
  sorry


end NUMINAMATH_CALUDE_suzanna_bike_ride_l4136_413613


namespace NUMINAMATH_CALUDE_win_sector_area_l4136_413657

/-- Given a circular spinner with radius 8 cm and a probability of winning 1/4,
    prove that the area of the WIN sector is 16π square centimeters. -/
theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 1/4) :
  p * π * r^2 = 16 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l4136_413657


namespace NUMINAMATH_CALUDE_max_profit_at_84_l4136_413626

/-- Defective rate as a function of daily output -/
def defective_rate (x : ℕ) : ℚ :=
  if x ≤ 94 then 1 / (96 - x) else 2/3

/-- Daily profit as a function of daily output and profit per qualified instrument -/
def daily_profit (x : ℕ) (A : ℚ) : ℚ :=
  if x ≤ 94 then
    (x * (1 - defective_rate x) * A) - (x * defective_rate x * (A/2))
  else 0

theorem max_profit_at_84 (A : ℚ) (h : A > 0) :
  ∀ x : ℕ, x ≠ 0 → daily_profit 84 A ≥ daily_profit x A :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_84_l4136_413626


namespace NUMINAMATH_CALUDE_medicine_price_reduction_l4136_413652

theorem medicine_price_reduction (x : ℝ) :
  (100 : ℝ) > 0 ∧ (81 : ℝ) > 0 →
  (∃ (initial_price final_price : ℝ),
    initial_price = 100 ∧
    final_price = 81 ∧
    final_price = initial_price * (1 - x) * (1 - x)) →
  100 * (1 - x)^2 = 81 :=
by sorry

end NUMINAMATH_CALUDE_medicine_price_reduction_l4136_413652


namespace NUMINAMATH_CALUDE_jason_potato_eating_time_l4136_413674

/-- Given that Jason eats 27 potatoes in 3 hours, prove that it takes him 20 minutes to eat 3 potatoes. -/
theorem jason_potato_eating_time :
  ∀ (total_potatoes total_hours potatoes_to_eat : ℕ) (minutes_per_hour : ℕ),
    total_potatoes = 27 →
    total_hours = 3 →
    potatoes_to_eat = 3 →
    minutes_per_hour = 60 →
    (potatoes_to_eat * total_hours * minutes_per_hour) / total_potatoes = 20 := by
  sorry

end NUMINAMATH_CALUDE_jason_potato_eating_time_l4136_413674


namespace NUMINAMATH_CALUDE_cost_is_five_l4136_413699

/-- The number of tickets available -/
def total_tickets : ℕ := 10

/-- The number of rides possible -/
def number_of_rides : ℕ := 2

/-- The cost of each ride in tickets -/
def cost_per_ride : ℕ := total_tickets / number_of_rides

/-- Theorem: The cost per ride is 5 tickets -/
theorem cost_is_five : cost_per_ride = 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_is_five_l4136_413699


namespace NUMINAMATH_CALUDE_negative_number_identification_l4136_413619

theorem negative_number_identification (a b c d : ℝ) : 
  a = -6 ∧ b = 0 ∧ c = 0.2 ∧ d = 3 →
  (a < 0 ∧ b ≥ 0 ∧ c > 0 ∧ d > 0) := by sorry

end NUMINAMATH_CALUDE_negative_number_identification_l4136_413619


namespace NUMINAMATH_CALUDE_product_digit_sum_l4136_413604

/-- The first 101-digit number -/
def number1 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

/-- The second 101-digit number -/
def number2 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

/-- Function to get the hundreds digit of a number -/
def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

/-- Function to get the tens digit of a number -/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem product_digit_sum :
  hundreds_digit (number1 * number2) + tens_digit (number1 * number2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l4136_413604


namespace NUMINAMATH_CALUDE_composite_probability_is_point_68_l4136_413658

/-- The number of natural numbers from 1 to 50 -/
def total_numbers : ℕ := 50

/-- The number of composite numbers from 1 to 50 -/
def composite_count : ℕ := 34

/-- The probability of selecting a composite number from the first 50 natural numbers -/
def composite_probability : ℚ := composite_count / total_numbers

/-- Theorem: The probability of selecting a composite number from the first 50 natural numbers is 0.68 -/
theorem composite_probability_is_point_68 : composite_probability = 68 / 100 := by
  sorry

end NUMINAMATH_CALUDE_composite_probability_is_point_68_l4136_413658


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l4136_413690

theorem like_terms_exponent_sum (m n : ℤ) : 
  (m + 2 = 6 ∧ n + 1 = 3) → (-m)^3 + n^2 = -60 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l4136_413690


namespace NUMINAMATH_CALUDE_circular_track_length_l4136_413698

theorem circular_track_length :
  ∀ (track_length : ℝ) (brenda_speed sally_speed : ℝ),
    brenda_speed > 0 →
    sally_speed > 0 →
    track_length / 2 - 120 = 120 * sally_speed / brenda_speed →
    track_length / 2 + 40 = (track_length / 2 - 80) * sally_speed / brenda_speed →
    track_length = 480 := by
  sorry

end NUMINAMATH_CALUDE_circular_track_length_l4136_413698


namespace NUMINAMATH_CALUDE_c_months_is_eleven_l4136_413625

/-- Represents the rental scenario for a pasture -/
structure PastureRental where
  total_rent : ℕ
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  b_months : ℕ
  c_horses : ℕ
  b_payment : ℕ

/-- Calculates the number of months c put in horses -/
def calculate_c_months (rental : PastureRental) : ℕ :=
  sorry

/-- Theorem stating that given the rental conditions, c put in horses for 11 months -/
theorem c_months_is_eleven (rental : PastureRental) 
  (h1 : rental.total_rent = 841)
  (h2 : rental.a_horses = 12)
  (h3 : rental.a_months = 8)
  (h4 : rental.b_horses = 16)
  (h5 : rental.b_months = 9)
  (h6 : rental.c_horses = 18)
  (h7 : rental.b_payment = 348) :
  calculate_c_months rental = 11 :=
by sorry

end NUMINAMATH_CALUDE_c_months_is_eleven_l4136_413625


namespace NUMINAMATH_CALUDE_solve_colored_copies_l4136_413656

def colored_copies_problem (colored_cost white_cost : ℚ) (total_copies : ℕ) (total_cost : ℚ) : Prop :=
  ∃ (colored_copies : ℕ),
    colored_copies ≤ total_copies ∧
    colored_cost * colored_copies + white_cost * (total_copies - colored_copies) = total_cost ∧
    colored_copies = 50

theorem solve_colored_copies :
  colored_copies_problem (10/100) (5/100) 400 (45/2) :=
sorry

end NUMINAMATH_CALUDE_solve_colored_copies_l4136_413656


namespace NUMINAMATH_CALUDE_erica_earnings_l4136_413675

/-- The amount of money earned per kilogram of fish -/
def price_per_kg : ℝ := 20

/-- The amount of fish caught in the past four months in kilograms -/
def past_four_months_catch : ℝ := 80

/-- The amount of fish caught today in kilograms -/
def today_catch : ℝ := 2 * past_four_months_catch

/-- The total amount of fish caught in kilograms -/
def total_catch : ℝ := past_four_months_catch + today_catch

/-- Erica's total earnings for the past four months including today -/
def total_earnings : ℝ := total_catch * price_per_kg

theorem erica_earnings : total_earnings = 4800 := by
  sorry

end NUMINAMATH_CALUDE_erica_earnings_l4136_413675


namespace NUMINAMATH_CALUDE_at_most_two_rational_points_l4136_413659

/-- A point in the 2D plane -/
structure Point where
  x : ℚ
  y : ℚ

/-- A circle in the 2D plane -/
structure Circle where
  center_x : ℚ
  center_y : ℝ
  radius : ℝ

/-- A point is on a circle if it satisfies the circle equation -/
def Point.onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center_x)^2 + (p.y - c.center_y)^2 = c.radius^2

/-- The main theorem: there are at most two rational points on a circle with irrational y-coordinate of the center -/
theorem at_most_two_rational_points (c : Circle) 
    (h : Irrational c.center_y) :
    ∃ (p1 p2 : Point), ∀ (p : Point), 
      p.onCircle c → p = p1 ∨ p = p2 := by
  sorry

end NUMINAMATH_CALUDE_at_most_two_rational_points_l4136_413659


namespace NUMINAMATH_CALUDE_A_work_time_l4136_413695

/-- The number of days it takes B to complete the work alone -/
def B_days : ℝ := 8

/-- The total payment for the work -/
def total_payment : ℝ := 3600

/-- The payment to C -/
def C_payment : ℝ := 450

/-- The number of days it takes A, B, and C to complete the work together -/
def combined_days : ℝ := 3

/-- The number of days it takes A to complete the work alone -/
def A_days : ℝ := 56

theorem A_work_time :
  ∃ (C_rate : ℝ),
    (1 / A_days + 1 / B_days + C_rate = 1 / combined_days) ∧
    (1 / A_days : ℝ) / (1 / B_days) = (total_payment - C_payment) / C_payment :=
by sorry

end NUMINAMATH_CALUDE_A_work_time_l4136_413695


namespace NUMINAMATH_CALUDE_hot_chocolate_consumption_l4136_413640

/-- The number of cups of hot chocolate John drinks in 5 hours -/
def cups_in_five_hours : ℕ := 15

/-- The time interval between each cup of hot chocolate in minutes -/
def interval : ℕ := 20

/-- The total time in minutes -/
def total_time : ℕ := 5 * 60

theorem hot_chocolate_consumption :
  cups_in_five_hours = total_time / interval :=
by sorry

end NUMINAMATH_CALUDE_hot_chocolate_consumption_l4136_413640


namespace NUMINAMATH_CALUDE_more_polygons_without_A1_l4136_413665

-- Define the number of points on the circle
def n : ℕ := 16

-- Define the function to calculate the number of polygons including A1
def polygons_with_A1 (n : ℕ) : ℕ :=
  (2^(n-1) : ℕ) - (n : ℕ)

-- Define the function to calculate the number of polygons not including A1
def polygons_without_A1 (n : ℕ) : ℕ :=
  (2^(n-1) : ℕ) - (n : ℕ) - ((n-1).choose 2)

-- State the theorem
theorem more_polygons_without_A1 :
  polygons_without_A1 n > polygons_with_A1 n :=
by sorry

end NUMINAMATH_CALUDE_more_polygons_without_A1_l4136_413665


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_m_eq_neg_three_l4136_413694

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of a real parameter m. -/
def z (m : ℝ) : ℂ :=
  Complex.mk (m^2 + m - 6) (m - 2)

/-- If z(m) is purely imaginary, then m = -3. -/
theorem purely_imaginary_implies_m_eq_neg_three :
  ∀ m : ℝ, is_purely_imaginary (z m) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_m_eq_neg_three_l4136_413694


namespace NUMINAMATH_CALUDE_chess_game_draw_probability_l4136_413600

theorem chess_game_draw_probability (p_win : ℝ) (p_not_lose : ℝ) 
  (h1 : p_win = 0.4) 
  (h2 : p_not_lose = 0.9) : 
  p_not_lose - p_win = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_draw_probability_l4136_413600


namespace NUMINAMATH_CALUDE_choir_members_count_l4136_413651

theorem choir_members_count : ∃! n : ℕ, 
  200 ≤ n ∧ n ≤ 300 ∧ 
  n % 10 = 4 ∧ 
  n % 11 = 5 ∧ 
  n = 234 := by sorry

end NUMINAMATH_CALUDE_choir_members_count_l4136_413651


namespace NUMINAMATH_CALUDE_sixth_edge_possibilities_l4136_413661

/-- Represents the edge lengths of a tetrahedron -/
structure TetrahedronEdges :=
  (a b c d e f : ℕ)

/-- Checks if three lengths satisfy the triangle inequality -/
def satisfiesTriangleInequality (x y z : ℕ) : Prop :=
  x + y > z ∧ y + z > x ∧ z + x > y

/-- Checks if all faces of a tetrahedron satisfy the triangle inequality -/
def validTetrahedron (t : TetrahedronEdges) : Prop :=
  satisfiesTriangleInequality t.a t.b t.c ∧
  satisfiesTriangleInequality t.a t.d t.e ∧
  satisfiesTriangleInequality t.b t.d t.f ∧
  satisfiesTriangleInequality t.c t.e t.f

/-- The main theorem stating that there are exactly 6 possible lengths for the sixth edge -/
theorem sixth_edge_possibilities :
  ∃! (s : Finset ℕ),
    s.card = 6 ∧
    (∀ x, x ∈ s ↔ ∃ t : TetrahedronEdges,
      t.a = 14 ∧ t.b = 20 ∧ t.c = 40 ∧ t.d = 52 ∧ t.e = 70 ∧ t.f = x ∧
      validTetrahedron t) :=
by sorry


end NUMINAMATH_CALUDE_sixth_edge_possibilities_l4136_413661


namespace NUMINAMATH_CALUDE_sandbag_weight_l4136_413617

/-- Calculates the weight of a partially filled sandbag with a heavier material -/
theorem sandbag_weight (capacity : ℝ) (fill_percentage : ℝ) (weight_increase : ℝ) : 
  capacity = 450 →
  fill_percentage = 0.75 →
  weight_increase = 0.65 →
  capacity * fill_percentage * (1 + weight_increase) = 556.875 := by
  sorry

end NUMINAMATH_CALUDE_sandbag_weight_l4136_413617


namespace NUMINAMATH_CALUDE_equal_numbers_product_l4136_413620

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 →
  a = 12 →
  b = 22 →
  c = d →
  c * d = 529 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l4136_413620


namespace NUMINAMATH_CALUDE_sum_ab_equals_four_l4136_413668

theorem sum_ab_equals_four (a b c d : ℤ) 
  (h1 : b + c = 7) 
  (h2 : c + d = 5) 
  (h3 : a + d = 2) : 
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_ab_equals_four_l4136_413668


namespace NUMINAMATH_CALUDE_trace_bag_weight_l4136_413645

theorem trace_bag_weight (trace_bags : ℕ) (gordon_bags : ℕ) (gordon_bag1_weight : ℕ) (gordon_bag2_weight : ℕ) :
  trace_bags = 5 →
  gordon_bags = 2 →
  gordon_bag1_weight = 3 →
  gordon_bag2_weight = 7 →
  (trace_bags : ℚ) * (trace_bags.div gordon_bags) * (gordon_bag1_weight + gordon_bag2_weight) = trace_bags * 2 :=
by sorry

end NUMINAMATH_CALUDE_trace_bag_weight_l4136_413645


namespace NUMINAMATH_CALUDE_intersection_M_N_l4136_413622

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

-- Define set N
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (4 - x^2)}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Icc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4136_413622


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4136_413627

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt 32 + Real.sqrt x = Real.sqrt 50 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4136_413627
