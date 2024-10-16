import Mathlib

namespace NUMINAMATH_CALUDE_no_integer_solutions_to_equation_l117_11722

theorem no_integer_solutions_to_equation :
  ¬∃ (w x y z : ℤ), (5 : ℝ)^w + (5 : ℝ)^x = (7 : ℝ)^y + (7 : ℝ)^z :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_to_equation_l117_11722


namespace NUMINAMATH_CALUDE_total_cost_is_39_47_l117_11735

def marbles_cost : Float := 9.05
def football_cost : Float := 4.95
def baseball_cost : Float := 6.52
def toy_car_original_cost : Float := 6.50
def toy_car_discount_percent : Float := 20
def puzzle_cost : Float := 3.25
def puzzle_quantity : Nat := 2
def action_figure_discounted_cost : Float := 10.50

def calculate_discounted_price (original_price : Float) (discount_percent : Float) : Float :=
  original_price * (1 - discount_percent / 100)

def calculate_total_cost : Float :=
  marbles_cost +
  football_cost +
  baseball_cost +
  calculate_discounted_price toy_car_original_cost toy_car_discount_percent +
  puzzle_cost +
  action_figure_discounted_cost

theorem total_cost_is_39_47 :
  calculate_total_cost = 39.47 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_39_47_l117_11735


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_A_in_U_l117_11712

-- Define the universal set U
def U : Set ℝ := {x | 1 < x ∧ x < 7}

-- Define set A
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | 3 ≤ x ∧ x < 5} := by sorry

-- Theorem for complement of A in U
theorem complement_A_in_U : (U \ A) = {x | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 7)} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_A_in_U_l117_11712


namespace NUMINAMATH_CALUDE_square_diagonal_cut_l117_11771

theorem square_diagonal_cut (s : ℝ) (h : s = 10) : 
  let diagonal := s * Real.sqrt 2
  ∃ (a b c : ℝ), a = s ∧ b = s ∧ c = diagonal ∧ 
    a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_cut_l117_11771


namespace NUMINAMATH_CALUDE_remainder_of_n_l117_11751

theorem remainder_of_n (n : ℕ) 
  (h1 : n^2 % 7 = 3) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_n_l117_11751


namespace NUMINAMATH_CALUDE_area_between_curves_l117_11728

-- Define the two curves
def curve1 (x : ℝ) : ℝ := x^3 - x
def curve2 (a x : ℝ) : ℝ := x^2 - a

-- Define the derivatives of the curves
def curve1_derivative (x : ℝ) : ℝ := 3 * x^2 - 1
def curve2_derivative (x : ℝ) : ℝ := 2 * x

-- Theorem statement
theorem area_between_curves :
  ∃ (a : ℝ) (P : ℝ × ℝ),
    -- Conditions:
    -- 1. P lies on both curves
    curve1 P.1 = P.2 ∧
    curve2 a P.1 = P.2 ∧
    -- 2. The curves have a common tangent at P
    curve1_derivative P.1 = curve2_derivative P.1 →
    -- Conclusion:
    -- The area between the curves is 13/12
    (∫ x in (Real.sqrt 5 / 2 - 1 / 6)..(1 / 6 + Real.sqrt 5 / 2), |curve1 x - curve2 a x|) = 13 / 12 :=
by
  sorry

end NUMINAMATH_CALUDE_area_between_curves_l117_11728


namespace NUMINAMATH_CALUDE_spider_dressing_combinations_l117_11762

/-- The number of legs of the spider -/
def num_legs : ℕ := 10

/-- The number of socks per leg -/
def socks_per_leg : ℕ := 2

/-- The number of shoes per leg -/
def shoes_per_leg : ℕ := 1

/-- The total number of items to wear -/
def total_items : ℕ := num_legs * (socks_per_leg + shoes_per_leg)

/-- The number of ways to arrange socks on one leg -/
def sock_arrangements_per_leg : ℕ := 2  -- 2! = 2

theorem spider_dressing_combinations :
  (Nat.choose total_items num_legs) * (sock_arrangements_per_leg ^ num_legs) =
  (Nat.factorial total_items) / (Nat.factorial num_legs * Nat.factorial (total_items - num_legs)) * 1024 :=
by sorry

end NUMINAMATH_CALUDE_spider_dressing_combinations_l117_11762


namespace NUMINAMATH_CALUDE_unique_geometric_progression_pair_l117_11741

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricProgression (x y z w : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ y = x * r ∧ z = y * r ∧ w = z * r

/-- There exists exactly one pair of real numbers (a, b) such that 12, a, b, ab form a geometric progression. -/
theorem unique_geometric_progression_pair :
  ∃! (a b : ℝ), IsGeometricProgression 12 a b (a * b) := by
  sorry

#check unique_geometric_progression_pair

end NUMINAMATH_CALUDE_unique_geometric_progression_pair_l117_11741


namespace NUMINAMATH_CALUDE_letters_in_mailboxes_l117_11763

/-- The number of ways to distribute n items into k categories -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of letters -/
def num_letters : ℕ := 4

/-- The number of mailboxes -/
def num_mailboxes : ℕ := 3

/-- Theorem: The number of ways to put 4 letters into 3 mailboxes is 81 -/
theorem letters_in_mailboxes :
  distribute num_letters num_mailboxes = 81 := by sorry

end NUMINAMATH_CALUDE_letters_in_mailboxes_l117_11763


namespace NUMINAMATH_CALUDE_battery_current_l117_11733

/-- Given a battery with voltage 48V, prove that the current I is 4A when the resistance R is 12Ω. -/
theorem battery_current (I R : ℝ) : 
  (∀ R, I = 48 / R) →  -- The relationship between I and R for any R
  R = 12 →             -- The specific resistance value
  I = 4 :=             -- The current to be proved
by sorry

end NUMINAMATH_CALUDE_battery_current_l117_11733


namespace NUMINAMATH_CALUDE_not_blessed_2017_l117_11744

def is_valid_date (month day : ℕ) : Prop :=
  1 ≤ month ∧ month ≤ 12 ∧ 1 ≤ day ∧ day ≤ 31

def concat_mmdd (month day : ℕ) : ℕ :=
  month * 100 + day

def is_blessed_year (year : ℕ) : Prop :=
  ∃ (month day : ℕ), is_valid_date month day ∧ concat_mmdd month day = year % 100

theorem not_blessed_2017 : ¬ is_blessed_year 2017 :=
sorry

end NUMINAMATH_CALUDE_not_blessed_2017_l117_11744


namespace NUMINAMATH_CALUDE_regular_icosahedron_faces_l117_11783

/-- A regular icosahedron is a polyhedron with identical equilateral triangular faces. -/
structure RegularIcosahedron where
  is_polyhedron : Bool
  has_identical_equilateral_triangular_faces : Bool

/-- The number of faces of a regular icosahedron is 20. -/
theorem regular_icosahedron_faces (i : RegularIcosahedron) : Nat :=
  20

#check regular_icosahedron_faces

end NUMINAMATH_CALUDE_regular_icosahedron_faces_l117_11783


namespace NUMINAMATH_CALUDE_six_boxes_consecutive_green_balls_l117_11781

/-- The number of ways to fill n boxes with red or green balls, such that at least one box
    contains a green ball and the boxes containing green balls are consecutively numbered. -/
def consecutiveGreenBalls (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

/-- Theorem stating that for 6 boxes, there are 21 ways to fill them under the given conditions. -/
theorem six_boxes_consecutive_green_balls :
  consecutiveGreenBalls 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_six_boxes_consecutive_green_balls_l117_11781


namespace NUMINAMATH_CALUDE_intersection_and_midpoint_trajectory_l117_11726

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y-1)^2 = 5

-- Define the line l
def line_l (m x y : ℝ) : Prop := m*x - y + 1 - m = 0

-- Define the trajectory of midpoint M
def trajectory_M (x y : ℝ) : Prop := (x - 1/2)^2 + (y-1)^2 = 1/4

theorem intersection_and_midpoint_trajectory :
  ∀ m : ℝ,
  (∃ A B : ℝ × ℝ, A ≠ B ∧ circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l m A.1 A.2 ∧ line_l m B.1 B.2) ∧
  (∀ x y : ℝ, (∃ A B : ℝ × ℝ, A ≠ B ∧ circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l m A.1 A.2 ∧ line_l m B.1 B.2 ∧
    x = (A.1 + B.1) / 2 ∧ y = (A.2 + B.2) / 2) → trajectory_M x y) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_midpoint_trajectory_l117_11726


namespace NUMINAMATH_CALUDE_luis_gum_contribution_l117_11724

/-- Calculates the number of gum pieces Luis gave to Maria -/
def luisGumPieces (initialPieces tomsContribution totalPieces : ℕ) : ℕ :=
  totalPieces - (initialPieces + tomsContribution)

theorem luis_gum_contribution :
  luisGumPieces 25 16 61 = 20 := by
  sorry

end NUMINAMATH_CALUDE_luis_gum_contribution_l117_11724


namespace NUMINAMATH_CALUDE_tangent_and_minimum_value_l117_11796

open Real

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := exp x * (a * x^2 + b * x + 1)

-- Define the derivative of f
noncomputable def f' (a b x : ℝ) : ℝ := exp x * (a * x^2 + (2 * a + b) * x + b + 1)

theorem tangent_and_minimum_value (a b : ℝ) :
  (f' a b (-1) = 0) →
  (
    -- Part I
    (b = 1 →
      ∃ (m c : ℝ), m = 2 ∧ c = 1 ∧
      ∀ x y, y = f a b x ∧ x = 0 → y = m * x + c
    ) ∧
    -- Part II
    (
      (∀ x, x ∈ Set.Icc (-1) 1 → f a b x ≥ 0) ∧
      (∃ x, x ∈ Set.Icc (-1) 1 ∧ f a b x = 0) →
      b = 2 ∨ b = -2
    )
  ) := by sorry

end NUMINAMATH_CALUDE_tangent_and_minimum_value_l117_11796


namespace NUMINAMATH_CALUDE_age_ratio_problem_l117_11718

/-- Given Tom's current age t and Lily's current age l, prove that the smallest positive integer x
    that satisfies (t + x) / (l + x) = 3 is 22, where t and l satisfy the given conditions. -/
theorem age_ratio_problem (t l : ℕ) (h1 : t - 3 = 5 * (l - 3)) (h2 : t - 8 = 6 * (l - 8)) :
  (∃ x : ℕ, x > 0 ∧ (t + x : ℚ) / (l + x) = 3 ∧ ∀ y : ℕ, y > 0 → (t + y : ℚ) / (l + y) = 3 → x ≤ y) →
  (∃ x : ℕ, x = 22 ∧ x > 0 ∧ (t + x : ℚ) / (l + x) = 3 ∧ ∀ y : ℕ, y > 0 → (t + y : ℚ) / (l + y) = 3 → x ≤ y) :=
by
  sorry


end NUMINAMATH_CALUDE_age_ratio_problem_l117_11718


namespace NUMINAMATH_CALUDE_dress_price_calculation_l117_11752

/-- Given a dress with an original price, discount rate, and tax rate, 
    calculate the total selling price after discount and tax. -/
def totalSellingPrice (originalPrice : ℝ) (discountRate : ℝ) (taxRate : ℝ) : ℝ :=
  let salePrice := originalPrice * (1 - discountRate)
  let taxAmount := salePrice * taxRate
  salePrice + taxAmount

/-- Theorem stating that for a dress with original price $80, 25% discount, 
    and 10% tax, the total selling price is $66. -/
theorem dress_price_calculation :
  totalSellingPrice 80 0.25 0.10 = 66 := by
  sorry

#eval totalSellingPrice 80 0.25 0.10

end NUMINAMATH_CALUDE_dress_price_calculation_l117_11752


namespace NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l117_11747

theorem contrapositive_square_sum_zero (m n : ℝ) :
  (¬(mn = 0) → ¬(m^2 + n^2 = 0)) ↔ (m^2 + n^2 = 0 → mn = 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l117_11747


namespace NUMINAMATH_CALUDE_total_payment_example_l117_11780

/-- Calculates the total amount paid for a meal including sales tax and tip -/
def total_payment (meal_cost : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  meal_cost * (1 + sales_tax_rate + tip_rate)

/-- Theorem: The total payment for a $100 meal with 4% sales tax and 6% tip is $110 -/
theorem total_payment_example : total_payment 100 0.04 0.06 = 110 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_example_l117_11780


namespace NUMINAMATH_CALUDE_multiple_condition_l117_11785

theorem multiple_condition (n : ℕ) : 
  n = 1475 → 0 < n → n < 2006 → ∃ k : ℕ, 2006 * n = k * (2006 + n) :=
sorry

end NUMINAMATH_CALUDE_multiple_condition_l117_11785


namespace NUMINAMATH_CALUDE_not_all_odd_l117_11711

theorem not_all_odd (a b c d : ℕ) (h1 : a = b * c + d) (h2 : d < b) : 
  ¬(Odd a ∧ Odd b ∧ Odd c ∧ Odd d) := by
  sorry

end NUMINAMATH_CALUDE_not_all_odd_l117_11711


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l117_11710

theorem no_real_roots_quadratic (b : ℝ) : ∀ x : ℝ, x^2 - b*x + 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l117_11710


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l117_11799

theorem absolute_value_equation_product (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (|5 * x₁| - 7 = 28) ∧ (|5 * x₂| - 7 = 28) ∧ (x₁ ≠ x₂) ∧ (x₁ * x₂ = -49)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l117_11799


namespace NUMINAMATH_CALUDE_tan_neg_seven_pi_sixths_l117_11707

theorem tan_neg_seven_pi_sixths : 
  Real.tan (-7 * π / 6) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_seven_pi_sixths_l117_11707


namespace NUMINAMATH_CALUDE_function_expressions_and_minimum_l117_11708

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x * (x + 2)
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b * x + 2

def has_same_tangent_at_zero (f g : ℝ → ℝ) : Prop :=
  (deriv f) 0 = (deriv g) 0 ∧ f 0 = g 0

theorem function_expressions_and_minimum (a b : ℝ) (t : ℝ) 
  (h1 : has_same_tangent_at_zero (f a) (g b))
  (h2 : t > -4) :
  (∃ (a' b' : ℝ), f a' = f 1 ∧ g b' = g 3) ∧
  (∀ x ∈ Set.Icc t (t + 1),
    (t < -3 → f 1 x ≥ -Real.exp (-3)) ∧
    (t ≥ -3 → f 1 x ≥ Real.exp t * (t + 2))) :=
by sorry

end NUMINAMATH_CALUDE_function_expressions_and_minimum_l117_11708


namespace NUMINAMATH_CALUDE_cafeteria_red_apples_l117_11784

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 42

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 7

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 9

/-- The number of extra apples -/
def extra_apples : ℕ := 40

/-- Theorem: The cafeteria ordered 42 red apples -/
theorem cafeteria_red_apples :
  red_apples = 42 ∧
  red_apples + green_apples = students_wanting_fruit + extra_apples :=
sorry

end NUMINAMATH_CALUDE_cafeteria_red_apples_l117_11784


namespace NUMINAMATH_CALUDE_pauls_crayons_left_l117_11701

/-- Represents the number of crayons Paul had left at the end of the school year. -/
def crayons_left (initial_erasers initial_crayons : ℕ) (extra_crayons : ℕ) : ℕ :=
  initial_erasers + extra_crayons

/-- Theorem stating that Paul had 523 crayons left at the end of the school year. -/
theorem pauls_crayons_left :
  crayons_left 457 617 66 = 523 := by
  sorry

end NUMINAMATH_CALUDE_pauls_crayons_left_l117_11701


namespace NUMINAMATH_CALUDE_julians_comic_frames_l117_11721

/-- The number of frames on each page of Julian's comic book -/
def frames_per_page : ℕ := 11

/-- The number of pages in Julian's comic book -/
def total_pages : ℕ := 13

/-- The total number of frames in Julian's comic book -/
def total_frames : ℕ := frames_per_page * total_pages

theorem julians_comic_frames :
  total_frames = 143 := by
  sorry

end NUMINAMATH_CALUDE_julians_comic_frames_l117_11721


namespace NUMINAMATH_CALUDE_reflection_sequence_exists_l117_11702

/-- Definition of a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Definition of a triangle using three points -/
structure Triangle :=
  (p1 : Point)
  (p2 : Point)
  (p3 : Point)

/-- Definition of a reflection line -/
inductive ReflectionLine
  | AB
  | BC
  | CA

/-- A sequence of reflections -/
def ReflectionSequence := List ReflectionLine

/-- Apply a single reflection to a point -/
def reflect (p : Point) (line : ReflectionLine) : Point :=
  match line with
  | ReflectionLine.AB => ⟨p.x, -p.y⟩
  | ReflectionLine.BC => ⟨3 - p.y, 3 - p.x⟩
  | ReflectionLine.CA => ⟨-p.x, p.y⟩

/-- Apply a sequence of reflections to a point -/
def applyReflections (p : Point) (seq : ReflectionSequence) : Point :=
  seq.foldl reflect p

/-- Apply a sequence of reflections to a triangle -/
def reflectTriangle (t : Triangle) (seq : ReflectionSequence) : Triangle :=
  ⟨applyReflections t.p1 seq, applyReflections t.p2 seq, applyReflections t.p3 seq⟩

/-- The original triangle -/
def originalTriangle : Triangle :=
  ⟨⟨0, 0⟩, ⟨0, 1⟩, ⟨2, 0⟩⟩

/-- The target triangle -/
def targetTriangle : Triangle :=
  ⟨⟨24, 36⟩, ⟨24, 37⟩, ⟨26, 36⟩⟩

theorem reflection_sequence_exists : ∃ (seq : ReflectionSequence), reflectTriangle originalTriangle seq = targetTriangle := by
  sorry

end NUMINAMATH_CALUDE_reflection_sequence_exists_l117_11702


namespace NUMINAMATH_CALUDE_smart_mart_puzzles_sold_l117_11720

/-- The number of science kits sold by Smart Mart last week -/
def science_kits : ℕ := 45

/-- The difference between science kits and puzzles sold -/
def difference : ℕ := 9

/-- The number of puzzles sold by Smart Mart last week -/
def puzzles : ℕ := science_kits - difference

theorem smart_mart_puzzles_sold : puzzles = 36 := by
  sorry

end NUMINAMATH_CALUDE_smart_mart_puzzles_sold_l117_11720


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l117_11704

def I : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 3, 4}
def B : Set Nat := {3, 4, 5, 6}

theorem complement_intersection_theorem :
  (A ∩ B)ᶜ = {1, 2, 5, 6} :=
by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l117_11704


namespace NUMINAMATH_CALUDE_short_sleeve_shirts_count_l117_11782

/-- The number of short sleeve shirts washed -/
def short_sleeve_shirts : ℕ := 9 - 5

/-- The total number of shirts washed -/
def total_shirts : ℕ := 9

/-- The number of long sleeve shirts washed -/
def long_sleeve_shirts : ℕ := 5

theorem short_sleeve_shirts_count : short_sleeve_shirts = 4 := by
  sorry

end NUMINAMATH_CALUDE_short_sleeve_shirts_count_l117_11782


namespace NUMINAMATH_CALUDE_flower_bed_max_area_l117_11789

/-- The maximum area of a rectangular flower bed with given constraints -/
theorem flower_bed_max_area : 
  ∀ l w : ℝ, 
  l = 150 → 
  l + 2*w = 450 → 
  0 < l → 
  0 < w → 
  l * w ≤ 22500 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_max_area_l117_11789


namespace NUMINAMATH_CALUDE_problem_statement_l117_11700

theorem problem_statement : |Real.sqrt 3 - 2| + 2 * Real.sin (60 * π / 180) - 2023^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l117_11700


namespace NUMINAMATH_CALUDE_common_root_inequality_l117_11774

theorem common_root_inequality (a b t : ℝ) (ha : a > 0) (hb : b > 0) (ht : t > 1)
  (eq1 : t^2 + a*t - 100 = 0) (eq2 : t^2 - 200*t + b = 0) : b - a > 100 := by
  sorry

end NUMINAMATH_CALUDE_common_root_inequality_l117_11774


namespace NUMINAMATH_CALUDE_sum_of_angles_is_360_l117_11714

-- Define the angles
variable (A B C D E F : ℝ)

-- Define the triangles and quadrilateral
def triangle_ABC := A + B + C = 180
def triangle_DEF := D + E + F = 180
def quadrilateral_BEFC := B + E + F + C = 360

-- State the theorem
theorem sum_of_angles_is_360 
  (h1 : triangle_ABC A B C) 
  (h2 : triangle_DEF D E F) 
  (h3 : quadrilateral_BEFC B E F C) : 
  A + B + C + D + E + F = 360 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_is_360_l117_11714


namespace NUMINAMATH_CALUDE_min_value_sum_product_l117_11730

theorem min_value_sum_product (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l117_11730


namespace NUMINAMATH_CALUDE_f_properties_l117_11795

-- Define the function f(x) = x^2 - 2x + 1
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Theorem stating the properties of f(x)
theorem f_properties :
  (∃ x : ℝ, f x = 0 ∧ x = 1) ∧
  (f 0 * f 2 > 0) ∧
  (¬ ∀ x y : ℝ, x < y → x < 0 → f x > f y) ∧
  (∀ x : ℝ, x < 0 → f x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l117_11795


namespace NUMINAMATH_CALUDE_no_valid_coloring_l117_11709

/-- Represents a coloring of a 4x4 grid -/
def Coloring := Fin 4 → Fin 4 → Fin 8

/-- Checks if two cells are adjacent in a 4x4 grid -/
def adjacent (r1 c1 r2 c2 : Fin 4) : Prop :=
  (r1 = r2 ∧ (c1 = c2 + 1 ∨ c2 = c1 + 1)) ∨
  (c1 = c2 ∧ (r1 = r2 + 1 ∨ r2 = r1 + 1))

/-- Checks if a coloring satisfies the condition that every pair of colors
    appears on adjacent cells -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ color1 color2 : Fin 8, color1 < color2 →
    ∃ r1 c1 r2 c2 : Fin 4, 
      adjacent r1 c1 r2 c2 ∧
      c r1 c1 = color1 ∧ c r2 c2 = color2

/-- The main theorem stating that no valid coloring exists -/
theorem no_valid_coloring : ¬∃ c : Coloring, valid_coloring c := by
  sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l117_11709


namespace NUMINAMATH_CALUDE_multiply_72_68_l117_11791

theorem multiply_72_68 : 72 * 68 = 4896 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_multiply_72_68_l117_11791


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l117_11758

theorem polynomial_evaluation (x : ℝ) (hx_pos : x > 0) (hx_eq : x^2 - 3*x - 9 = 0) :
  x^4 - 3*x^3 - 9*x^2 + 27*x - 8 = -8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l117_11758


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l117_11757

/-- A geometric sequence with common ratio q -/
def geometric_sequence (q : ℝ) : ℕ → ℝ := fun n => q ^ (n - 1)

/-- The common ratio of a geometric sequence where a₄ = 27 and a₇ = -729 -/
theorem geometric_sequence_ratio : ∃ q : ℝ, 
  geometric_sequence q 4 = 27 ∧ 
  geometric_sequence q 7 = -729 ∧ 
  q = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l117_11757


namespace NUMINAMATH_CALUDE_angle2_value_l117_11776

-- Define the angles
variable (angle1 angle2 angle3 : ℝ)

-- Define the conditions
def complementary (a b : ℝ) : Prop := a + b = 90
def supplementary (a b : ℝ) : Prop := a + b = 180

-- State the theorem
theorem angle2_value (h1 : complementary angle1 angle2)
                     (h2 : supplementary angle1 angle3)
                     (h3 : angle3 = 125) :
  angle2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_angle2_value_l117_11776


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l117_11794

/-- Given an ellipse mx^2 + y^2 = 1 with eccentricity √3/2, its major axis length is either 2 or 4 -/
theorem ellipse_major_axis_length (m : ℝ) :
  (∃ (x y : ℝ), m * x^2 + y^2 = 1) →  -- Ellipse equation
  (∃ (a b : ℝ), a > b ∧ a^2 * m = b^2 ∧ (a^2 - b^2) / a^2 = 3/4) →  -- Eccentricity condition
  (∃ (l : ℝ), l = 2 ∨ l = 4 ∧ l = 2 * a) :=  -- Major axis length
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l117_11794


namespace NUMINAMATH_CALUDE_common_root_values_l117_11775

theorem common_root_values (a b c d k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hk1 : a * k^4 + b * k^3 + c * k^2 + d * k + a = 0)
  (hk2 : b * k^4 + c * k^3 + d * k^2 + a * k + b = 0) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_common_root_values_l117_11775


namespace NUMINAMATH_CALUDE_emma_age_l117_11727

def guesses : List Nat := [26, 29, 31, 33, 35, 39, 42, 44, 47, 50]

def is_prime (n : Nat) : Prop := Nat.Prime n

def off_by_one (guess : Nat) (age : Nat) : Prop :=
  guess = age - 1 ∨ guess = age + 1

def count_lower_guesses (age : Nat) : Nat :=
  guesses.filter (· < age) |>.length

theorem emma_age : ∃ (age : Nat),
  age ∈ guesses ∧
  is_prime age ∧
  (count_lower_guesses age : Rat) / guesses.length ≥ 6/10 ∧
  (∃ (g1 g2 : Nat), g1 ∈ guesses ∧ g2 ∈ guesses ∧ g1 ≠ g2 ∧ 
    off_by_one g1 age ∧ off_by_one g2 age) ∧
  age = 43 := by
  sorry

end NUMINAMATH_CALUDE_emma_age_l117_11727


namespace NUMINAMATH_CALUDE_intersection_implies_a_values_l117_11768

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + (a^2-5) = 0}

theorem intersection_implies_a_values (a : ℝ) : A ∩ B a = {2} → a = -1 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_values_l117_11768


namespace NUMINAMATH_CALUDE_union_A_complement_B_equals_result_l117_11779

-- Define the set I
def I : Set ℤ := {x | |x| < 3}

-- Define set A
def A : Set ℤ := {1, 2}

-- Define set B
def B : Set ℤ := {-2, -1, 2}

-- Theorem statement
theorem union_A_complement_B_equals_result : A ∪ (I \ B) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_A_complement_B_equals_result_l117_11779


namespace NUMINAMATH_CALUDE_sixth_term_is_27_eighth_term_is_46_l117_11754

-- First sequence
def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem sixth_term_is_27 : arithmetic_sequence 2 5 6 = 27 := by sorry

-- Second sequence
def even_indexed_term (n : ℕ) : ℕ → ℕ
  | 0 => 4
  | m + 1 => 2 * even_indexed_term n m + 1

def odd_indexed_term (n : ℕ) : ℕ → ℕ
  | 0 => 2
  | m + 1 => 2 * odd_indexed_term n m + 2

def combined_sequence (n : ℕ) : ℕ :=
  if n % 2 = 0 then even_indexed_term (n / 2) (n / 2 - 1)
  else odd_indexed_term ((n + 1) / 2) ((n - 1) / 2)

theorem eighth_term_is_46 : combined_sequence 8 = 46 := by sorry

end NUMINAMATH_CALUDE_sixth_term_is_27_eighth_term_is_46_l117_11754


namespace NUMINAMATH_CALUDE_sin_n_squared_not_converge_to_zero_l117_11790

theorem sin_n_squared_not_converge_to_zero :
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (N : ℕ), ∃ (n : ℕ), n > N ∧ |Real.sin (n^2 : ℝ)| ≥ ε :=
sorry

end NUMINAMATH_CALUDE_sin_n_squared_not_converge_to_zero_l117_11790


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l117_11731

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Theorem statement
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + a 2 = 1 →
  a 3 + a 4 = 4 →
  a 5 + a 6 + a 7 + a 8 = 80 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l117_11731


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_l117_11729

/-- The coefficient of x^3 in the expansion of (3x^3 + 2x^2 + 5x + 3)(4x^3 + 5x^2 + 6x + 8) is 61 -/
theorem coefficient_x_cubed (x : ℝ) : 
  let p₁ : Polynomial ℝ := 3 * X^3 + 2 * X^2 + 5 * X + 3
  let p₂ : Polynomial ℝ := 4 * X^3 + 5 * X^2 + 6 * X + 8
  (p₁ * p₂).coeff 3 = 61 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_l117_11729


namespace NUMINAMATH_CALUDE_one_is_hilbert_number_h_hilbert_formula_larger_h_hilbert_number_l117_11739

-- Definition of a Hilbert number
def is_hilbert_number (p : ℕ) : Prop :=
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ p = x^2 + y^2 - x*y

-- Definition of an H Hilbert number
def is_h_hilbert_number (p : ℕ) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ p = (2*n - 1)^2 + (2*n + 1)^2 - (2*n - 1)*(2*n + 1)

-- Theorem statements
theorem one_is_hilbert_number : is_hilbert_number 1 := by sorry

theorem h_hilbert_formula (n : ℕ) (h : n > 0) : 
  is_h_hilbert_number (4*n^2 + 3) := by sorry

theorem larger_h_hilbert_number (m n : ℕ) (hm : m > 0) (hn : n > 0) (h_diff : 4*n^2 + 3 - (4*m^2 + 3) = 48) :
  4*n^2 + 3 = 67 := by sorry

end NUMINAMATH_CALUDE_one_is_hilbert_number_h_hilbert_formula_larger_h_hilbert_number_l117_11739


namespace NUMINAMATH_CALUDE_externally_tangent_case_intersecting_case_l117_11759

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4
def center_O₂ : ℝ × ℝ := (2, 1)

-- Define the equations for O₂
def equation_O₂_tangent (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 12 - 8 * Real.sqrt 2
def equation_O₂_intersect_1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4
def equation_O₂_intersect_2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 20

-- Theorem for externally tangent case
theorem externally_tangent_case :
  (∀ x y, circle_O₁ x y → ¬equation_O₂_tangent x y) ∧
  (∃ x y, circle_O₁ x y ∧ equation_O₂_tangent x y) →
  ∀ x y, equation_O₂_tangent x y :=
sorry

-- Theorem for intersecting case
theorem intersecting_case (A B : ℝ × ℝ) :
  (A ≠ B) ∧
  (∀ x y, circle_O₁ x y ↔ ((x - A.1)^2 + (y - A.2)^2 = 0 ∨ (x - B.1)^2 + (y - B.2)^2 = 0)) ∧
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 8) →
  (∀ x y, equation_O₂_intersect_1 x y ∨ equation_O₂_intersect_2 x y) :=
sorry

end NUMINAMATH_CALUDE_externally_tangent_case_intersecting_case_l117_11759


namespace NUMINAMATH_CALUDE_ladder_problem_l117_11717

/-- Given a right triangle with hypotenuse 13 meters and one leg 12 meters,
    prove that the other leg is 5 meters. -/
theorem ladder_problem (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : b = 12) :
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_ladder_problem_l117_11717


namespace NUMINAMATH_CALUDE_spatial_relationships_l117_11778

structure Space3D where
  Point : Type
  Line : Type
  Plane : Type
  intersects : Line → Plane → Prop
  parallel : Line → Line → Prop
  parallel_line_plane : Line → Plane → Prop
  perpendicular : Line → Line → Prop
  in_plane : Line → Plane → Prop

variable (S : Space3D)

theorem spatial_relationships :
  (∀ (a : S.Line) (α : S.Plane), S.intersects a α → ¬∃ (l : S.Line), S.in_plane l α ∧ S.parallel l a) ∧
  (∃ (a b : S.Line) (α : S.Plane), S.parallel_line_plane b α ∧ S.perpendicular a b ∧ S.parallel_line_plane a α) ∧
  (∃ (a b : S.Line) (α : S.Plane), S.parallel a b ∧ S.in_plane b α ∧ ¬S.parallel_line_plane a α) :=
by sorry

end NUMINAMATH_CALUDE_spatial_relationships_l117_11778


namespace NUMINAMATH_CALUDE_extended_triangle_similarity_l117_11746

/-- Triangle ABC with given side lengths --/
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)
  (CA : ℝ)

/-- Extended triangle PABC --/
structure ExtendedTriangle extends Triangle :=
  (PC : ℝ)

/-- Similarity of triangles PAB and PCA --/
def is_similar (t : ExtendedTriangle) : Prop :=
  t.PC / t.AB = t.CA / t.PC

theorem extended_triangle_similarity (t : ExtendedTriangle) 
  (h1 : t.AB = 8)
  (h2 : t.BC = 7)
  (h3 : t.CA = 6)
  (h4 : is_similar t) :
  t.PC = 9 := by
  sorry

end NUMINAMATH_CALUDE_extended_triangle_similarity_l117_11746


namespace NUMINAMATH_CALUDE_sum_first_100_base6_l117_11767

/-- Represents a number in base 6 --/
def Base6 := Nat

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : Nat) : Base6 := sorry

/-- Adds two numbers in base 6 --/
def addBase6 (a b : Base6) : Base6 := sorry

/-- Multiplies two numbers in base 6 --/
def mulBase6 (a b : Base6) : Base6 := sorry

/-- Divides two numbers in base 6 --/
def divBase6 (a b : Base6) : Base6 := sorry

/-- Computes the sum of the first n (in base 6) natural numbers in base 6 --/
def sumFirstNBase6 (n : Base6) : Base6 := sorry

theorem sum_first_100_base6 :
  sumFirstNBase6 (toBase6 100) = toBase6 7222 := by sorry

end NUMINAMATH_CALUDE_sum_first_100_base6_l117_11767


namespace NUMINAMATH_CALUDE_dinos_third_gig_rate_l117_11760

/-- Dino's monthly income calculation -/
def monthly_income (hours1 hours2 hours3 : ℕ) (rate1 rate2 rate3 : ℚ) : ℚ :=
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3

/-- Theorem: Dino's hourly rate for the third gig is $40/hour -/
theorem dinos_third_gig_rate :
  ∀ (rate3 : ℚ),
  monthly_income 20 30 5 10 20 rate3 = 1000 →
  rate3 = 40 := by
sorry

end NUMINAMATH_CALUDE_dinos_third_gig_rate_l117_11760


namespace NUMINAMATH_CALUDE_children_count_l117_11756

theorem children_count (pencils_per_child : ℕ) (total_pencils : ℕ) (h1 : pencils_per_child = 6) (h2 : total_pencils = 12) :
  total_pencils / pencils_per_child = 2 :=
by sorry

end NUMINAMATH_CALUDE_children_count_l117_11756


namespace NUMINAMATH_CALUDE_remainder_3_304_mod_11_l117_11761

theorem remainder_3_304_mod_11 : 3^304 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_304_mod_11_l117_11761


namespace NUMINAMATH_CALUDE_triangle_area_sum_l117_11769

-- Define points on a line
variable (A B C D E : ℝ)

-- Define lengths
variable (AB BC CD : ℝ)

-- Define areas
variable (S_MAC S_NBC S_MCD S_NCE : ℝ)

-- State the theorem
theorem triangle_area_sum :
  A < B ∧ B < C ∧ C < D ∧ D < E →  -- Points are on the same line in order
  AB = 4 →
  BC = 3 →
  CD = 2 →
  S_MAC + S_NBC = 51 →
  S_MCD + S_NCE = 32 →
  S_MCD + S_NBC = 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_sum_l117_11769


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l117_11742

-- Define the universal set U as ℝ
def U := ℝ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 + x + 1 ≥ 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x ≥ 3}

-- Define the complement of B with respect to U
def C_U_B : Set ℝ := {x : ℝ | x ∉ B}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ C_U_B = {x : ℝ | x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l117_11742


namespace NUMINAMATH_CALUDE_square_circle_union_area_l117_11703

/-- The area of the union of a square with side length 8 and a circle with radius 8
    centered at one of the square's vertices is equal to 64 + 48π. -/
theorem square_circle_union_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 8
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  let overlap_area := (1 / 4) * circle_area
  square_area + circle_area - overlap_area = 64 + 48 * π := by
sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l117_11703


namespace NUMINAMATH_CALUDE_garden_area_not_covered_by_flower_beds_l117_11764

def garden_side_length : ℝ := 16
def flower_bed_radius : ℝ := 8

theorem garden_area_not_covered_by_flower_beds :
  let total_area := garden_side_length ^ 2
  let flower_bed_area := 4 * (π * flower_bed_radius ^ 2) / 4
  total_area - flower_bed_area = 256 - 64 * π := by sorry

end NUMINAMATH_CALUDE_garden_area_not_covered_by_flower_beds_l117_11764


namespace NUMINAMATH_CALUDE_distribute_4_3_l117_11719

/-- The number of ways to distribute n identical objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: Distributing 4 identical balls into 3 different boxes can be done in 15 ways -/
theorem distribute_4_3 : distribute 4 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_distribute_4_3_l117_11719


namespace NUMINAMATH_CALUDE_smallest_distance_between_complex_circles_l117_11740

theorem smallest_distance_between_complex_circles
  (z w : ℂ)
  (hz : Complex.abs (z - (2 + 2*Complex.I)) = 2)
  (hw : Complex.abs (w + (3 + 5*Complex.I)) = 4) :
  Complex.abs (z - w) ≥ Real.sqrt 74 - 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_complex_circles_l117_11740


namespace NUMINAMATH_CALUDE_greater_number_is_84_l117_11736

theorem greater_number_is_84 (x y : ℝ) (h1 : x * y = 2688) (h2 : (x + y) - (x - y) = 64) : max x y = 84 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_is_84_l117_11736


namespace NUMINAMATH_CALUDE_min_fraction_sum_l117_11798

def Digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem min_fraction_sum (W X Y Z : ℕ) 
  (hw : W ∈ Digits) (hx : X ∈ Digits) (hy : Y ∈ Digits) (hz : Z ∈ Digits)
  (hdiff : W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z) :
  (∀ W' X' Y' Z' : ℕ, 
    W' ∈ Digits → X' ∈ Digits → Y' ∈ Digits → Z' ∈ Digits →
    W' ≠ X' ∧ W' ≠ Y' ∧ W' ≠ Z' ∧ X' ≠ Y' ∧ X' ≠ Z' ∧ Y' ≠ Z' →
    (W : ℚ) / X + (Y : ℚ) / Z ≤ (W' : ℚ) / X' + (Y' : ℚ) / Z') →
  (W : ℚ) / X + (Y : ℚ) / Z = 23 / 21 := by
  sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l117_11798


namespace NUMINAMATH_CALUDE_apple_ratio_problem_l117_11738

theorem apple_ratio_problem (green_apples red_apples : ℕ) : 
  (green_apples : ℚ) / red_apples = 5 / 3 → 
  green_apples = 15 → 
  red_apples = 9 := by
sorry

end NUMINAMATH_CALUDE_apple_ratio_problem_l117_11738


namespace NUMINAMATH_CALUDE_tomato_egg_soup_min_time_l117_11753

/-- Represents a cooking step with its duration -/
structure CookingStep where
  name : String
  duration : ℕ

/-- The set of cooking steps for Tomato Egg Soup -/
def tomatoEggSoupSteps : List CookingStep := [
  ⟨"A", 1⟩,
  ⟨"B", 2⟩,
  ⟨"C", 3⟩,
  ⟨"D", 1⟩,
  ⟨"E", 1⟩
]

/-- Calculates the minimum time required to complete all cooking steps -/
def minCookingTime (steps : List CookingStep) : ℕ := sorry

/-- Theorem: The minimum time to make Tomato Egg Soup is 6 minutes -/
theorem tomato_egg_soup_min_time :
  minCookingTime tomatoEggSoupSteps = 6 := by sorry

end NUMINAMATH_CALUDE_tomato_egg_soup_min_time_l117_11753


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l117_11777

theorem smallest_solution_of_equation :
  let f (x : ℝ) := (3 * x) / (x - 3) + (3 * x^2 - 27) / x
  ∃ (smallest : ℝ), smallest = (2 - Real.sqrt 31) / 3 ∧
    f smallest = 16 ∧
    ∀ (y : ℝ), y ≠ 3 ∧ y ≠ 0 ∧ f y = 16 → y ≥ smallest :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l117_11777


namespace NUMINAMATH_CALUDE_at_least_three_functional_probability_l117_11770

def num_lamps : ℕ := 5
def func_prob : ℝ := 0.2

theorem at_least_three_functional_probability :
  let p := func_prob
  let q := 1 - p
  let binom_prob (n k : ℕ) := (Nat.choose n k : ℝ) * p^k * q^(n-k)
  binom_prob num_lamps 3 + binom_prob num_lamps 4 + binom_prob num_lamps 5 = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_at_least_three_functional_probability_l117_11770


namespace NUMINAMATH_CALUDE_f_range_l117_11705

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem f_range : ∀ x : ℝ, f x = -3 * π / 4 ∨ f x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l117_11705


namespace NUMINAMATH_CALUDE_sum_of_fractions_l117_11750

theorem sum_of_fractions (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l117_11750


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l117_11748

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a > 0)
  (x₁ x₂ : ℝ)
  (hroots : ∀ x, f a b c x - x = 0 ↔ x = x₁ ∨ x = x₂)
  (horder : 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/a) :
  (∀ x, 0 < x ∧ x < x₁ → x < f a b c x ∧ f a b c x < x₁) ∧
  (-b / (2*a) < x₁ / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l117_11748


namespace NUMINAMATH_CALUDE_smallest_n_exceeding_100000_l117_11749

def sequence_term (n : ℕ) : ℕ := 9 + 10 * (n - 1)

def sequence_sum (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ i => sequence_term (i + 1))

theorem smallest_n_exceeding_100000 : 
  (∀ k < 142, sequence_sum k ≤ 100000) ∧ 
  sequence_sum 142 > 100000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_exceeding_100000_l117_11749


namespace NUMINAMATH_CALUDE_mean_temperature_is_87_l117_11706

def temperatures : List ℝ := [84, 86, 85, 87, 89, 90, 88]

theorem mean_temperature_is_87 :
  (temperatures.sum / temperatures.length : ℝ) = 87 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_87_l117_11706


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l117_11723

/-- Given an equilateral triangle where the area is twice the length of one of its sides,
    prove that its perimeter is 8√3 units. -/
theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) : 
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l117_11723


namespace NUMINAMATH_CALUDE_function_properties_l117_11743

/-- Given functions f and g satisfying certain properties, prove specific characteristics -/
theorem function_properties (f g : ℝ → ℝ) 
  (h1 : ∀ x y, f (x - y) = f x * g y - g x * f y)
  (h2 : f (-2) = f 1)
  (h3 : f 1 ≠ 0) : 
  (g 0 = 1) ∧ 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ (x : ℝ) (k : ℤ), f x = f (x + 3 * ↑k)) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l117_11743


namespace NUMINAMATH_CALUDE_smallest_divisor_after_subtraction_l117_11793

theorem smallest_divisor_after_subtraction (n m k : ℕ) (h1 : n = 899830) (h2 : m = 6) (h3 : k = 8) :
  k > m ∧
  (n - m) % k = 0 ∧
  ∀ d : ℕ, m < d ∧ d < k → (n - m) % d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_subtraction_l117_11793


namespace NUMINAMATH_CALUDE_colonization_ways_l117_11792

/-- Represents the number of Earth-like planets -/
def earth_like_planets : ℕ := 5

/-- Represents the number of Mars-like planets -/
def mars_like_planets : ℕ := 6

/-- Represents the units required to colonize an Earth-like planet -/
def earth_like_units : ℕ := 2

/-- Represents the units required to colonize a Mars-like planet -/
def mars_like_units : ℕ := 1

/-- Represents the total available units for colonization -/
def total_units : ℕ := 14

/-- Theorem stating that there are exactly 20 different ways to occupy the planets -/
theorem colonization_ways : 
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 ≤ earth_like_planets ∧ 
    p.2 ≤ mars_like_planets ∧ 
    p.1 * earth_like_units + p.2 * mars_like_units = total_units)
  (Finset.product (Finset.range (earth_like_planets + 1)) (Finset.range (mars_like_planets + 1)))).card = 20 :=
sorry

end NUMINAMATH_CALUDE_colonization_ways_l117_11792


namespace NUMINAMATH_CALUDE_project_hours_difference_l117_11788

theorem project_hours_difference (total_hours : ℕ) 
  (h_total : total_hours = 198) 
  (h_pat_kate : ∃ (kate : ℕ), pat = 2 * kate) 
  (h_pat_mark : ∃ (mark : ℕ), pat = mark / 3) : 
  ∃ (kate mark : ℕ), mark - kate = 110 ∧ 
    kate + pat + mark = total_hours := by
  sorry

end NUMINAMATH_CALUDE_project_hours_difference_l117_11788


namespace NUMINAMATH_CALUDE_students_not_in_biology_l117_11725

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880)
  (h2 : biology_percentage = 40 / 100) :
  ↑total_students * (1 - biology_percentage) = 528 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l117_11725


namespace NUMINAMATH_CALUDE_largest_whole_number_less_than_150_l117_11772

theorem largest_whole_number_less_than_150 :
  ∀ x : ℕ, x ≤ 24 ↔ 6 * x + 3 < 150 :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_less_than_150_l117_11772


namespace NUMINAMATH_CALUDE_divisible_by_1968_l117_11787

theorem divisible_by_1968 (n : ℕ) : ∃ k : ℤ, 
  (-1)^(2*n) + 9^(4*n) - 6^(8*n) + 8^(16*n) = 1968 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_1968_l117_11787


namespace NUMINAMATH_CALUDE_proportion_equality_false_l117_11797

theorem proportion_equality_false : 
  ¬(∀ (A B C : ℚ), (A / B = C / 4 ∧ A = 4) → B = C) :=
by sorry

end NUMINAMATH_CALUDE_proportion_equality_false_l117_11797


namespace NUMINAMATH_CALUDE_chef_michel_pies_l117_11716

/-- The number of pieces a shepherd's pie is cut into -/
def shepherds_pie_pieces : ℕ := 4

/-- The number of pieces a chicken pot pie is cut into -/
def chicken_pot_pie_pieces : ℕ := 5

/-- The number of customers who ordered shepherd's pie -/
def shepherds_pie_customers : ℕ := 52

/-- The number of customers who ordered chicken pot pie -/
def chicken_pot_pie_customers : ℕ := 80

/-- The total number of pies sold by Chef Michel -/
def total_pies_sold : ℕ := 
  (shepherds_pie_customers / shepherds_pie_pieces) + 
  (chicken_pot_pie_customers / chicken_pot_pie_pieces)

theorem chef_michel_pies : total_pies_sold = 29 := by
  sorry

end NUMINAMATH_CALUDE_chef_michel_pies_l117_11716


namespace NUMINAMATH_CALUDE_equation_has_two_distinct_real_roots_l117_11765

-- Define the new operation
def star_op (a b : ℝ) : ℝ := a^2 - a*b + b

-- Theorem statement
theorem equation_has_two_distinct_real_roots :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ star_op x₁ 3 = 5 ∧ star_op x₂ 3 = 5 :=
by sorry

end NUMINAMATH_CALUDE_equation_has_two_distinct_real_roots_l117_11765


namespace NUMINAMATH_CALUDE_modulus_of_z_l117_11715

theorem modulus_of_z (z : ℂ) (h : z * (Complex.I + 1) = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l117_11715


namespace NUMINAMATH_CALUDE_area_ratio_similar_triangles_l117_11734

/-- Given two similar triangles with areas S and S₁, and similarity coefficient k, 
    prove that the ratio of their areas is equal to the square of the similarity coefficient. -/
theorem area_ratio_similar_triangles (S S₁ k : ℝ) (a b a₁ b₁ α : ℝ) :
  S = (1 / 2) * a * b * Real.sin α →
  S₁ = (1 / 2) * a₁ * b₁ * Real.sin α →
  a₁ = k * a →
  b₁ = k * b →
  k > 0 →
  S₁ / S = k^2 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_similar_triangles_l117_11734


namespace NUMINAMATH_CALUDE_delivery_problem_l117_11773

theorem delivery_problem (total_bottles : ℕ) (cider_bottles : ℕ) (beer_bottles : ℕ) 
  (h1 : total_bottles = 180)
  (h2 : cider_bottles = 40)
  (h3 : beer_bottles = 80)
  (h4 : cider_bottles + beer_bottles < total_bottles) :
  (cider_bottles / 2) + (beer_bottles / 2) + ((total_bottles - cider_bottles - beer_bottles) / 2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_delivery_problem_l117_11773


namespace NUMINAMATH_CALUDE_infiniteNestedSqrtEqualThree_l117_11745

/-- The value of the infinite expression sqrt(6 + sqrt(6 + sqrt(6 + ...))) -/
noncomputable def infiniteNestedSqrt : ℝ :=
  Real.sqrt (6 + Real.sqrt (6 + Real.sqrt (6 + Real.sqrt 6)))

/-- Theorem stating that the infinite nested square root equals 3 -/
theorem infiniteNestedSqrtEqualThree : infiniteNestedSqrt = 3 := by
  sorry

end NUMINAMATH_CALUDE_infiniteNestedSqrtEqualThree_l117_11745


namespace NUMINAMATH_CALUDE_positive_numbers_l117_11737

theorem positive_numbers (a b c : ℝ) 
  (sum_positive : a + b + c > 0)
  (sum_products_positive : a * b + b * c + c * a > 0)
  (product_positive : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_l117_11737


namespace NUMINAMATH_CALUDE_power_of_64_equals_128_l117_11713

theorem power_of_64_equals_128 : (64 : ℝ) ^ (7/6) = 128 := by
  have h : 64 = 2^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_64_equals_128_l117_11713


namespace NUMINAMATH_CALUDE_quadratic_sum_r_s_l117_11732

/-- Given a quadratic equation 9x^2 - 36x - 81 = 0, when written in the form (x+r)^2 = s, 
    the sum of r and s is equal to 11. -/
theorem quadratic_sum_r_s (x r s : ℝ) : 
  (9 * x^2 - 36 * x - 81 = 0) → 
  ((x + r)^2 = s) → 
  (r + s = 11) := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_r_s_l117_11732


namespace NUMINAMATH_CALUDE_functional_eq_solution_l117_11766

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = max (f (x + y)) (f x * f y)

/-- The main theorem stating that any function satisfying the functional equation
    must be constant with values between 0 and 1, inclusive -/
theorem functional_eq_solution (f : ℝ → ℝ) (h : SatisfiesFunctionalEq f) :
    ∃ c : ℝ, (0 ≤ c ∧ c ≤ 1) ∧ (∀ x : ℝ, f x = c) :=
  sorry

end NUMINAMATH_CALUDE_functional_eq_solution_l117_11766


namespace NUMINAMATH_CALUDE_impossibility_of_three_similar_parts_l117_11755

theorem impossibility_of_three_similar_parts :
  ∀ (x : ℝ), x > 0 → ¬∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = x ∧
    (a ≤ Real.sqrt 2 * b ∧ b ≤ Real.sqrt 2 * a) ∧
    (b ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * b) ∧
    (a ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * a) :=
by
  sorry


end NUMINAMATH_CALUDE_impossibility_of_three_similar_parts_l117_11755


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l117_11786

theorem complex_equation_solutions (z : ℂ) : 
  z^3 + z = 2 * Complex.abs z^2 → 
  z = 0 ∨ z = 1 ∨ z = -1 + 2*Complex.I ∨ z = -1 - 2*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l117_11786
