import Mathlib

namespace tangent_circle_exists_l3644_364481

-- Define the types for points and circles
def Point := ℝ × ℝ
def Circle := Point × ℝ  -- Center and radius

-- Define a function to check if a point is on a circle
def is_on_circle (p : Point) (c : Circle) : Prop :=
  let (center, radius) := c
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

-- Define a function to check if two circles are tangent
def are_circles_tangent (c1 c2 : Circle) : Prop :=
  let (center1, radius1) := c1
  let (center2, radius2) := c2
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = (radius1 + radius2)^2

-- Theorem statement
theorem tangent_circle_exists (c1 c2 : Circle) (T : Point) 
  (h1 : is_on_circle T c1) : 
  ∃ (c : Circle), are_circles_tangent c c1 ∧ are_circles_tangent c c2 ∧ is_on_circle T c :=
sorry

end tangent_circle_exists_l3644_364481


namespace second_amount_equals_600_l3644_364446

/-- Calculate simple interest -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

/-- The problem statement -/
theorem second_amount_equals_600 :
  ∃ (P : ℝ),
    simple_interest 100 0.05 48 = simple_interest P 0.10 4 ∧
    P = 600 := by
  sorry

end second_amount_equals_600_l3644_364446


namespace expression_simplification_and_evaluation_l3644_364472

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 3 - 2
  ((x - 1) / (x - 2) + (2 * x - 8) / (x^2 - 4)) / (x + 5) = Real.sqrt 3 / 3 := by
sorry

end expression_simplification_and_evaluation_l3644_364472


namespace leftover_space_is_one_l3644_364432

-- Define the wall length
def wall_length : ℝ := 15

-- Define the desk length
def desk_length : ℝ := 2

-- Define the bookcase length
def bookcase_length : ℝ := 1.5

-- Define the function to calculate the space left over
def space_left_over (n : ℕ) : ℝ :=
  wall_length - (n * desk_length + n * bookcase_length)

-- Theorem statement
theorem leftover_space_is_one :
  ∃ n : ℕ, n > 0 ∧ 
    space_left_over n = 1 ∧
    ∀ m : ℕ, m > n → space_left_over m < 1 :=
  sorry

end leftover_space_is_one_l3644_364432


namespace degree_of_g_l3644_364411

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := -9 * x^4 + 2 * x^3 - 7 * x + 8

-- State the theorem
theorem degree_of_g (g : ℝ → ℝ) :
  (∃ (a b : ℝ), ∀ x, f x + g x = a * x + b) →  -- degree of f(x) + g(x) is 1
  (∃ (a b c d e : ℝ), a ≠ 0 ∧ ∀ x, g x = a * x^4 + b * x^3 + c * x^2 + d * x + e) :=  -- g(x) is a polynomial of degree 4
by sorry

end degree_of_g_l3644_364411


namespace sum_arithmetic_series_base8_l3644_364459

/-- Conversion from base 8 to base 10 -/
def base8ToBase10 (x : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 8 -/
def base10ToBase8 (x : ℕ) : ℕ := sorry

/-- Sum of arithmetic series in base 8 -/
def sumArithmeticSeriesBase8 (n a l : ℕ) : ℕ :=
  base10ToBase8 ((n * (base8ToBase10 a + base8ToBase10 l)) / 2)

theorem sum_arithmetic_series_base8 :
  sumArithmeticSeriesBase8 36 1 36 = 1056 := by sorry

end sum_arithmetic_series_base8_l3644_364459


namespace sequence_sum_equals_63_l3644_364469

theorem sequence_sum_equals_63 : 
  (Finset.range 9).sum (fun i => (i + 4) * (1 - 1 / (i + 2))) = 63 := by sorry

end sequence_sum_equals_63_l3644_364469


namespace solve_equation_l3644_364467

theorem solve_equation (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 26 → n = 5 := by
  sorry

end solve_equation_l3644_364467


namespace expected_distinct_faces_proof_l3644_364491

/-- A fair six-sided die is rolled six times. -/
def roll_die (n : ℕ) : Type := Fin 6 → Fin n

/-- The probability of a specific face not appearing in a single roll. -/
def prob_not_appear : ℚ := 5 / 6

/-- The expected number of distinct faces appearing in six rolls of a fair die. -/
def expected_distinct_faces : ℚ := (6^6 - 5^6) / 6^5

/-- Theorem stating that the expected number of distinct faces appearing when a fair
    six-sided die is rolled six times is equal to (6^6 - 5^6) / 6^5. -/
theorem expected_distinct_faces_proof :
  expected_distinct_faces = (6^6 - 5^6) / 6^5 :=
by sorry

end expected_distinct_faces_proof_l3644_364491


namespace candy_box_original_price_l3644_364482

/-- Given a candy box with an original price, which after a 25% increase becomes 10 pounds,
    prove that the original price was 8 pounds. -/
theorem candy_box_original_price (original_price : ℝ) : 
  (original_price * 1.25 = 10) → original_price = 8 := by
  sorry

end candy_box_original_price_l3644_364482


namespace gem_purchase_theorem_l3644_364452

/-- Proves that given the conditions of gem purchasing and bonuses, 
    the amount spent to obtain 30,000 gems is $250. -/
theorem gem_purchase_theorem (gems_per_dollar : ℕ) (bonus_rate : ℚ) (final_gems : ℕ) : 
  gems_per_dollar = 100 →
  bonus_rate = 1/5 →
  final_gems = 30000 →
  (final_gems : ℚ) / (gems_per_dollar : ℚ) / (1 + bonus_rate) = 250 := by
  sorry

end gem_purchase_theorem_l3644_364452


namespace polynomial_equality_sum_l3644_364429

theorem polynomial_equality_sum (a b c d : ℤ) : 
  (∀ x, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + x^3 - 2*x^2 + 17*x - 5) → 
  a + b + c + d = 5 := by
sorry

end polynomial_equality_sum_l3644_364429


namespace regular_27gon_trapezoid_l3644_364421

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides -/
def IsTrapezoid (a b c d : ℝ × ℝ) : Prop := sorry

/-- Main theorem: Among any 7 vertices of a regular 27-gon, 4 can be selected that form a trapezoid -/
theorem regular_27gon_trapezoid (P : RegularPolygon 27) 
  (S : Finset (Fin 27)) (hS : S.card = 7) : 
  ∃ (a b c d : Fin 27), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
    IsTrapezoid (P.vertices a) (P.vertices b) (P.vertices c) (P.vertices d) :=
sorry

end regular_27gon_trapezoid_l3644_364421


namespace alphametic_puzzle_unique_solution_l3644_364471

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the alphametic puzzle IDA + ME = ORA -/
def AlphameticPuzzle (I D A M E R O : Digit) : Prop :=
  (100 * I.val + 10 * D.val + A.val) + (10 * M.val + E.val) = 
  (100 * O.val + 10 * R.val + A.val)

/-- The main theorem stating that there exists a unique solution to the puzzle -/
theorem alphametic_puzzle_unique_solution : 
  ∃! (I D A M E R O : Digit), 
    AlphameticPuzzle I D A M E R O ∧ 
    I ≠ D ∧ I ≠ A ∧ I ≠ M ∧ I ≠ E ∧ I ≠ R ∧ I ≠ O ∧
    D ≠ A ∧ D ≠ M ∧ D ≠ E ∧ D ≠ R ∧ D ≠ O ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ R ∧ A ≠ O ∧
    M ≠ E ∧ M ≠ R ∧ M ≠ O ∧
    E ≠ R ∧ E ≠ O ∧
    R ≠ O ∧
    R.val = 0 :=
by sorry

end alphametic_puzzle_unique_solution_l3644_364471


namespace triangle_excircle_relation_l3644_364444

/-- Given a triangle ABC with sides a, b, c and excircle radii r_a, r_b, r_c opposite to vertices A, B, C respectively, 
    the sum of the squares of each side divided by the product of its opposite excircle radius and the sum of the other two radii equals 2. -/
theorem triangle_excircle_relation (a b c r_a r_b r_c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0) :
  a^2 / (r_a * (r_b + r_c)) + b^2 / (r_b * (r_c + r_a)) + c^2 / (r_c * (r_a + r_b)) = 2 := by
  sorry

end triangle_excircle_relation_l3644_364444


namespace square_root_calculations_l3644_364431

theorem square_root_calculations :
  (Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0) ∧
  (Real.sqrt 6 / Real.sqrt 18 * Real.sqrt 27 = 3) := by
  sorry

end square_root_calculations_l3644_364431


namespace tangent_circles_radius_l3644_364433

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (r1 r2 d : ℝ) : Prop := d = r1 + r2

theorem tangent_circles_radius (d r1 r2 : ℝ) :
  d = 8 → r1 = 3 → externally_tangent r1 r2 d → r2 = 5 := by
  sorry

end tangent_circles_radius_l3644_364433


namespace fold_paper_sum_l3644_364450

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if two points are symmetric about a given line -/
def areSymmetric (p1 p2 : Point) (l : Line) : Prop :=
  -- Definition of symmetry about a line
  sorry

/-- Finds the fold line given two pairs of symmetric points -/
def findFoldLine (p1 p2 p3 p4 : Point) : Line :=
  -- Definition to find the fold line
  sorry

/-- Main theorem -/
theorem fold_paper_sum (m n : ℝ) :
  let p1 : Point := ⟨0, 2⟩
  let p2 : Point := ⟨4, 0⟩
  let p3 : Point := ⟨9, 5⟩
  let p4 : Point := ⟨m, n⟩
  let foldLine := findFoldLine p1 p2 p3 p4
  areSymmetric p1 p2 foldLine ∧ areSymmetric p3 p4 foldLine →
  m + n = 10 := by
  sorry

end fold_paper_sum_l3644_364450


namespace translated_function_eq_l3644_364410

-- Define the original function
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 1

-- Define the translated function
def g (x : ℝ) : ℝ := f (x + 1) + 3

-- Theorem stating that the translated function is equal to 3x^2 - 1
theorem translated_function_eq (x : ℝ) : g x = 3 * x^2 - 1 := by
  sorry

end translated_function_eq_l3644_364410


namespace alla_boris_meeting_l3644_364498

/-- Represents the meeting point of Alla and Boris along a straight alley with lampposts -/
def meeting_point (total_lampposts : ℕ) (alla_position : ℕ) (boris_position : ℕ) : ℕ :=
  alla_position + (total_lampposts - alla_position - boris_position + 1) / 2

/-- Theorem stating that Alla and Boris meet at lamppost 163 under the given conditions -/
theorem alla_boris_meeting :
  let total_lampposts : ℕ := 400
  let alla_start : ℕ := 1
  let boris_start : ℕ := total_lampposts
  let alla_position : ℕ := 55
  let boris_position : ℕ := 321
  meeting_point total_lampposts alla_position boris_position = 163 :=
by sorry

end alla_boris_meeting_l3644_364498


namespace john_initial_plays_l3644_364434

/-- The number of acts in each play -/
def acts_per_play : ℕ := 5

/-- The number of wigs John wears per act -/
def wigs_per_act : ℕ := 2

/-- The cost of each wig in dollars -/
def cost_per_wig : ℕ := 5

/-- The selling price of each wig from the dropped play in dollars -/
def selling_price_per_wig : ℕ := 4

/-- The total amount John spent in dollars -/
def total_spent : ℕ := 110

/-- The number of plays John was initially performing in -/
def initial_plays : ℕ := 3

theorem john_initial_plays :
  initial_plays * (acts_per_play * wigs_per_act * cost_per_wig) -
  (acts_per_play * wigs_per_act * selling_price_per_wig) = total_spent :=
by sorry

end john_initial_plays_l3644_364434


namespace solve_exponential_equation_l3644_364487

theorem solve_exponential_equation :
  ∃ y : ℝ, (3 : ℝ) ^ (y + 3) = 81 ^ y ∧ y = 1 := by sorry

end solve_exponential_equation_l3644_364487


namespace arcsin_sin_equation_solutions_l3644_364447

theorem arcsin_sin_equation_solutions :
  let S := {x : ℝ | -3 * Real.pi / 2 ≤ x ∧ x ≤ 3 * Real.pi / 2 ∧ Real.arcsin (Real.sin x) = x / 3}
  S = {-3 * Real.pi, -Real.pi, 0, Real.pi, 3 * Real.pi} := by
  sorry

end arcsin_sin_equation_solutions_l3644_364447


namespace valid_pairs_l3644_364420

theorem valid_pairs : 
  ∀ m n : ℕ, 
    (∃ k : ℕ, m + 1 = n * k) ∧ 
    (∃ l : ℕ, n^2 - n + 1 = m * l) → 
    ((m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 3 ∧ n = 2)) := by
  sorry

end valid_pairs_l3644_364420


namespace thirty_day_month_equal_tuesdays_thursdays_l3644_364425

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Counts the number of occurrences of a specific day in a 30-day month starting from a given day -/
def countDayInMonth (startDay : DayOfWeek) (dayToCount : DayOfWeek) : Nat :=
  sorry

/-- Checks if a 30-day month starting from a given day has equal Tuesdays and Thursdays -/
def hasEqualTuesdaysThursdays (startDay : DayOfWeek) : Bool :=
  countDayInMonth startDay DayOfWeek.Tuesday = countDayInMonth startDay DayOfWeek.Thursday

/-- Counts the number of possible start days for a 30-day month with equal Tuesdays and Thursdays -/
def countValidStartDays : Nat :=
  sorry

theorem thirty_day_month_equal_tuesdays_thursdays :
  countValidStartDays = 4 :=
sorry

end thirty_day_month_equal_tuesdays_thursdays_l3644_364425


namespace equation_solution_pairs_l3644_364495

theorem equation_solution_pairs : 
  {(p, q) : ℕ × ℕ | (p + 1)^(p - 1) + (p - 1)^(p + 1) = q^q} = {(1, 1), (2, 2)} := by
  sorry

end equation_solution_pairs_l3644_364495


namespace point_on_line_value_l3644_364486

/-- A point lies on a line if it satisfies the line's equation -/
def PointOnLine (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

theorem point_on_line_value :
  ∀ x : ℝ, PointOnLine 1 4 4 1 x 8 → x = -3 := by
  sorry

end point_on_line_value_l3644_364486


namespace line_perpendicular_to_plane_l3644_364468

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (l m n : Line) (α : Plane) 
  (h1 : intersect l m)
  (h2 : parallel l α)
  (h3 : parallel m α)
  (h4 : perpendicular n l)
  (h5 : perpendicular n m) :
  perpendicularToPlane n α :=
sorry

end line_perpendicular_to_plane_l3644_364468


namespace jen_triple_flips_l3644_364416

/-- Represents the number of flips in a specific type of flip. -/
def flips_per_type (flip_type : String) : ℕ :=
  if flip_type = "double" then 2 else 3

/-- Represents the total number of flips performed by a gymnast. -/
def total_flips (completed_flips : ℕ) (flip_type : String) : ℕ :=
  completed_flips * flips_per_type flip_type

theorem jen_triple_flips (tyler_double_flips : ℕ) (h1 : tyler_double_flips = 12) :
  let tyler_total_flips := total_flips tyler_double_flips "double"
  let jen_total_flips := 2 * tyler_total_flips
  jen_total_flips / flips_per_type "triple" = 16 := by
  sorry

end jen_triple_flips_l3644_364416


namespace perfect_square_condition_l3644_364438

theorem perfect_square_condition (x y : ℕ) :
  ∃ (n : ℕ), (x + y)^2 + 3*x + y + 1 = n^2 ↔ x = y :=
sorry

end perfect_square_condition_l3644_364438


namespace weaving_increase_l3644_364493

/-- The sum of an arithmetic sequence with n terms, first term a, and common difference d -/
def arithmetic_sum (n : ℕ) (a d : ℚ) : ℚ := n * a + n * (n - 1) / 2 * d

/-- The problem of finding the daily increase in weaving -/
theorem weaving_increase (a₁ : ℚ) (n : ℕ) (S : ℚ) (h1 : a₁ = 5) (h2 : n = 30) (h3 : S = 390) :
  ∃ d : ℚ, arithmetic_sum n a₁ d = S ∧ d = 16/29 := by
  sorry

end weaving_increase_l3644_364493


namespace orange_distribution_l3644_364435

theorem orange_distribution (oranges_per_child : ℕ) (total_oranges : ℕ) (num_children : ℕ) : 
  oranges_per_child = 3 → 
  total_oranges = 12 → 
  num_children * oranges_per_child = total_oranges →
  num_children = 4 := by
sorry

end orange_distribution_l3644_364435


namespace exists_irrational_between_3_and_4_l3644_364408

theorem exists_irrational_between_3_and_4 : ∃ x : ℝ, Irrational x ∧ 3 < x ∧ x < 4 := by
  sorry

end exists_irrational_between_3_and_4_l3644_364408


namespace mixture_weight_l3644_364443

/-- Calculates the weight of a mixture of two brands of vegetable ghee -/
theorem mixture_weight 
  (weight_a : ℝ) 
  (weight_b : ℝ) 
  (ratio_a : ℝ) 
  (ratio_b : ℝ) 
  (total_volume : ℝ) 
  (h1 : weight_a = 900) 
  (h2 : weight_b = 850) 
  (h3 : ratio_a = 3) 
  (h4 : ratio_b = 2) 
  (h5 : total_volume = 4) : 
  (weight_a * (ratio_a / (ratio_a + ratio_b)) * total_volume + 
   weight_b * (ratio_b / (ratio_a + ratio_b)) * total_volume) / 1000 = 3.52 := by
sorry

end mixture_weight_l3644_364443


namespace product_of_exponents_l3644_364466

theorem product_of_exponents (p r s : ℕ) : 
  3^p + 3^3 = 36 → 2^r + 18 = 50 → 5^s + 7^2 = 1914 → p * r * s = 40 := by
  sorry

end product_of_exponents_l3644_364466


namespace green_blue_difference_l3644_364424

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ
  color_sum : blue + yellow + green = total
  ratio : blue * 18 = total * 3 ∧ yellow * 18 = total * 7 ∧ green * 18 = total * 8

theorem green_blue_difference (bag : DiskBag) (h : bag.total = 144) :
  bag.green - bag.blue = 40 := by
  sorry

end green_blue_difference_l3644_364424


namespace complex_fraction_real_implies_zero_l3644_364407

theorem complex_fraction_real_implies_zero (a : ℝ) : 
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (((a : ℂ) + 2 * Complex.I) / ((a : ℂ) - 2 * Complex.I)).im = 0 →
  a = 0 := by sorry

end complex_fraction_real_implies_zero_l3644_364407


namespace x_range_l3644_364479

theorem x_range (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y) (h3 : y ≤ 7) :
  1 ≤ x ∧ x < 5/4 := by
  sorry

end x_range_l3644_364479


namespace birthday_gift_contribution_l3644_364437

theorem birthday_gift_contribution (total_cost boss_contribution num_remaining_employees : ℕ) 
  (h1 : total_cost = 100)
  (h2 : boss_contribution = 15)
  (h3 : num_remaining_employees = 5) :
  let todd_contribution := 2 * boss_contribution
  let remaining_cost := total_cost - todd_contribution - boss_contribution
  remaining_cost / num_remaining_employees = 11 := by
sorry

end birthday_gift_contribution_l3644_364437


namespace animal_sightings_sum_l3644_364423

/-- The number of animal sightings in January -/
def january_sightings : ℕ := 26

/-- The number of animal sightings in February -/
def february_sightings : ℕ := 3 * january_sightings

/-- The number of animal sightings in March -/
def march_sightings : ℕ := february_sightings / 2

/-- The number of animal sightings in April -/
def april_sightings : ℕ := 2 * march_sightings

/-- The total number of animal sightings over the four months -/
def total_sightings : ℕ := january_sightings + february_sightings + march_sightings + april_sightings

theorem animal_sightings_sum : total_sightings = 221 := by
  sorry

end animal_sightings_sum_l3644_364423


namespace expression_evaluation_l3644_364478

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(y+1) * y^(x-1)) / (y^y * x^x) = x^(y-x+1) * y^(x-y-1) := by
  sorry

end expression_evaluation_l3644_364478


namespace is_circle_center_l3644_364430

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 5 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, 1)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 10 :=
by sorry

end is_circle_center_l3644_364430


namespace paint_usage_correct_l3644_364412

/-- Represents the amount of paint used for a canvas size -/
structure PaintUsage where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total paint used for a given canvas size and count -/
def totalPaintUsed (usage : PaintUsage) (count : ℕ) : PaintUsage :=
  { red := usage.red * count
  , blue := usage.blue * count
  , yellow := usage.yellow * count
  , green := usage.green * count
  }

/-- Adds two PaintUsage structures -/
def addPaintUsage (a b : PaintUsage) : PaintUsage :=
  { red := a.red + b.red
  , blue := a.blue + b.blue
  , yellow := a.yellow + b.yellow
  , green := a.green + b.green
  }

theorem paint_usage_correct : 
  let extraLarge : PaintUsage := { red := 5, blue := 3, yellow := 2, green := 1 }
  let large : PaintUsage := { red := 4, blue := 2, yellow := 3, green := 1 }
  let medium : PaintUsage := { red := 3, blue := 1, yellow := 2, green := 1 }
  let small : PaintUsage := { red := 1, blue := 1, yellow := 1, green := 1 }
  
  let totalUsage := addPaintUsage
    (addPaintUsage
      (addPaintUsage
        (totalPaintUsed extraLarge 3)
        (totalPaintUsed large 5))
      (totalPaintUsed medium 6))
    (totalPaintUsed small 8)

  totalUsage.red = 61 ∧
  totalUsage.blue = 33 ∧
  totalUsage.yellow = 41 ∧
  totalUsage.green = 22 :=
by sorry


end paint_usage_correct_l3644_364412


namespace min_value_reciprocal_sum_l3644_364403

theorem min_value_reciprocal_sum (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  1/x + 4/y + 9/z ≥ 36 := by
sorry

end min_value_reciprocal_sum_l3644_364403


namespace reciprocal_roots_quadratic_equation_l3644_364417

theorem reciprocal_roots_quadratic_equation :
  ∀ (α β : ℝ),
  (α^2 - 7*α - 1 = 0) →
  (β^2 - 7*β - 1 = 0) →
  (α + β = 7) →
  (α * β = -1) →
  ((1/α)^2 + 7*(1/α) - 1 = 0) ∧
  ((1/β)^2 + 7*(1/β) - 1 = 0) :=
by sorry

end reciprocal_roots_quadratic_equation_l3644_364417


namespace binary_arithmetic_l3644_364413

-- Define binary numbers as natural numbers
def bin_10110 : ℕ := 22  -- 10110 in binary is 22 in decimal
def bin_1011 : ℕ := 11   -- 1011 in binary is 11 in decimal
def bin_11100 : ℕ := 28  -- 11100 in binary is 28 in decimal
def bin_11101 : ℕ := 29  -- 11101 in binary is 29 in decimal
def bin_100010 : ℕ := 34 -- 100010 in binary is 34 in decimal

-- Define a function to convert a natural number to its binary representation
def to_binary (n : ℕ) : List ℕ := sorry

-- Theorem statement
theorem binary_arithmetic :
  to_binary (bin_10110 + bin_1011 - bin_11100 + bin_11101) = to_binary bin_100010 :=
sorry

end binary_arithmetic_l3644_364413


namespace solution_to_equation_l3644_364463

theorem solution_to_equation :
  ∃! (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ (5 * x)^10 = (10 * y)^5 - 25 * x ∧ x = 1/5 ∧ y = 1 := by
  sorry

end solution_to_equation_l3644_364463


namespace quadratic_roots_distinct_l3644_364451

theorem quadratic_roots_distinct (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 8 = 0 ∧ x₂^2 + m*x₂ - 8 = 0 := by
  sorry

end quadratic_roots_distinct_l3644_364451


namespace quadratic_point_ordering_l3644_364418

/-- A quadratic function f(x) = (x+1)² + 1 -/
def f (x : ℝ) : ℝ := (x + 1)^2 + 1

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (-3, f (-3))

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (0, f 0)

/-- Point C on the graph of f -/
def C : ℝ × ℝ := (2, f 2)

theorem quadratic_point_ordering :
  B.2 < A.2 ∧ A.2 < C.2 := by sorry

end quadratic_point_ordering_l3644_364418


namespace louisa_travel_speed_l3644_364405

/-- The average speed of Louisa's travel -/
def average_speed : ℝ := 37.5

/-- The distance traveled on the first day -/
def distance_day1 : ℝ := 375

/-- The distance traveled on the second day -/
def distance_day2 : ℝ := 525

/-- The time difference between the two trips -/
def time_difference : ℝ := 4

theorem louisa_travel_speed :
  (distance_day2 / average_speed) = (distance_day1 / average_speed) + time_difference :=
sorry

end louisa_travel_speed_l3644_364405


namespace factorization_problem_1_l3644_364462

theorem factorization_problem_1 (a x : ℝ) : 3*a*x^2 - 6*a*x + 3*a = 3*a*(x-1)^2 := by sorry

end factorization_problem_1_l3644_364462


namespace restaurant_peppers_total_weight_l3644_364484

theorem restaurant_peppers_total_weight 
  (green_peppers : ℝ) 
  (red_peppers : ℝ) 
  (h1 : green_peppers = 0.3333333333333333) 
  (h2 : red_peppers = 0.3333333333333333) : 
  green_peppers + red_peppers = 0.6666666666666666 := by
sorry

end restaurant_peppers_total_weight_l3644_364484


namespace pictures_in_first_album_l3644_364439

theorem pictures_in_first_album (total_pictures : ℕ) (albums : ℕ) (pictures_per_album : ℕ) :
  total_pictures = 35 →
  albums = 3 →
  pictures_per_album = 7 →
  total_pictures - (albums * pictures_per_album) = 14 := by
  sorry

end pictures_in_first_album_l3644_364439


namespace f1_extrema_f2_extrema_l3644_364400

-- Function 1
def f1 (x : ℝ) : ℝ := x^3 + 2*x

theorem f1_extrema :
  (∃ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∀ x ∈ Set.Icc (-1 : ℝ) 1, f1 x ≥ f1 x₁) ∧
  (∃ x₂ ∈ Set.Icc (-1 : ℝ) 1, ∀ x ∈ Set.Icc (-1 : ℝ) 1, f1 x ≤ f1 x₂) ∧
  (f1 x₁ = -3) ∧ (f1 x₂ = 3) :=
sorry

-- Function 2
def f2 (x : ℝ) : ℝ := (x - 1) * (x - 2)^2

theorem f2_extrema :
  (∃ x₁ ∈ Set.Icc 0 3, ∀ x ∈ Set.Icc 0 3, f2 x ≥ f2 x₁) ∧
  (∃ x₂ ∈ Set.Icc 0 3, ∀ x ∈ Set.Icc 0 3, f2 x ≤ f2 x₂) ∧
  (f2 x₁ = -4) ∧ (f2 x₂ = 2) :=
sorry

end f1_extrema_f2_extrema_l3644_364400


namespace min_distance_between_curves_l3644_364476

/-- The minimum distance between a point on y = (1/2)e^x and a point on y = ln(2x) -/
theorem min_distance_between_curves : ∃ (min_dist : ℝ),
  min_dist = Real.sqrt 2 * (1 - Real.log 2) ∧
  ∀ (x₁ x₂ : ℝ),
    let p := (x₁, (1/2) * Real.exp x₁)
    let q := (x₂, Real.log (2 * x₂))
    Real.sqrt ((x₁ - x₂)^2 + ((1/2) * Real.exp x₁ - Real.log (2 * x₂))^2) ≥ min_dist :=
by sorry

end min_distance_between_curves_l3644_364476


namespace ram_pairs_sold_l3644_364464

/-- Represents the sales and earnings of a hardware store for a week. -/
structure StoreSales where
  graphics_cards : Nat
  hard_drives : Nat
  cpus : Nat
  ram_pairs : Nat
  graphics_card_price : Nat
  hard_drive_price : Nat
  cpu_price : Nat
  ram_pair_price : Nat
  total_earnings : Nat

/-- Calculates the total earnings from a given StoreSales. -/
def calculate_earnings (sales : StoreSales) : Nat :=
  sales.graphics_cards * sales.graphics_card_price +
  sales.hard_drives * sales.hard_drive_price +
  sales.cpus * sales.cpu_price +
  sales.ram_pairs * sales.ram_pair_price

/-- Theorem stating that the number of RAM pairs sold is 4. -/
theorem ram_pairs_sold (sales : StoreSales) :
  sales.graphics_cards = 10 →
  sales.hard_drives = 14 →
  sales.cpus = 8 →
  sales.graphics_card_price = 600 →
  sales.hard_drive_price = 80 →
  sales.cpu_price = 200 →
  sales.ram_pair_price = 60 →
  sales.total_earnings = 8960 →
  calculate_earnings sales = sales.total_earnings →
  sales.ram_pairs = 4 := by
  sorry


end ram_pairs_sold_l3644_364464


namespace probability_both_win_is_one_third_l3644_364428

/-- Represents the three types of lottery tickets -/
inductive Ticket
  | FirstPrize
  | SecondPrize
  | NonPrize

/-- Represents a draw of two tickets without replacement -/
def Draw := (Ticket × Ticket)

/-- The set of all possible draws -/
def allDraws : Finset Draw := sorry

/-- Predicate to check if a draw results in both people winning a prize -/
def bothWinPrize (draw : Draw) : Prop := 
  draw.1 ≠ Ticket.NonPrize ∧ draw.2 ≠ Ticket.NonPrize

/-- The set of draws where both people win a prize -/
def winningDraws : Finset Draw := sorry

/-- The probability of both people winning a prize -/
def probabilityBothWin : ℚ := (winningDraws.card : ℚ) / (allDraws.card : ℚ)

theorem probability_both_win_is_one_third : 
  probabilityBothWin = 1 / 3 := by sorry

end probability_both_win_is_one_third_l3644_364428


namespace sum_coordinates_of_D_l3644_364442

/-- Given that N(6,2) is the midpoint of line segment CD and C(10,-2), 
    prove that the sum of coordinates of D is 8 -/
theorem sum_coordinates_of_D (N C D : ℝ × ℝ) : 
  N = (6, 2) → 
  C = (10, -2) → 
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 8 := by
sorry

end sum_coordinates_of_D_l3644_364442


namespace fiftieth_term_of_sequence_l3644_364453

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem fiftieth_term_of_sequence :
  let a₁ := 3
  let d := 6
  let n := 50
  arithmetic_sequence a₁ d n = 297 := by sorry

end fiftieth_term_of_sequence_l3644_364453


namespace smallest_irrational_distance_points_theorem_l3644_364454

/-- The smallest number of points in ℝⁿ such that every point of ℝⁿ is an irrational distance from at least one of the points -/
def smallest_irrational_distance_points (n : ℕ) : ℕ :=
  if n = 1 then 2 else 3

/-- Theorem stating the smallest number of points in ℝⁿ such that every point of ℝⁿ is an irrational distance from at least one of the points -/
theorem smallest_irrational_distance_points_theorem (n : ℕ) (hn : n > 0) :
  smallest_irrational_distance_points n = if n = 1 then 2 else 3 :=
sorry

end smallest_irrational_distance_points_theorem_l3644_364454


namespace sum_of_binary_numbers_l3644_364470

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number 101(2) -/
def binary1 : List Bool := [true, false, true]

/-- The second binary number 110(2) -/
def binary2 : List Bool := [false, true, true]

/-- Statement: The sum of the decimal representations of 101(2) and 110(2) is 11 -/
theorem sum_of_binary_numbers :
  binary_to_decimal binary1 + binary_to_decimal binary2 = 11 := by
  sorry

end sum_of_binary_numbers_l3644_364470


namespace sum_of_squares_l3644_364461

theorem sum_of_squares (m n : ℝ) (h1 : m + n = 7) (h2 : m * n = 3) : m^2 + n^2 = 43 := by
  sorry

end sum_of_squares_l3644_364461


namespace alyssa_chicken_nuggets_l3644_364494

/-- Given 100 total chicken nuggets and two people eating twice as much as Alyssa,
    prove that Alyssa ate 20 chicken nuggets. -/
theorem alyssa_chicken_nuggets :
  ∀ (total : ℕ) (alyssa : ℕ),
    total = 100 →
    total = alyssa + 2 * alyssa + 2 * alyssa →
    alyssa = 20 :=
by
  sorry

end alyssa_chicken_nuggets_l3644_364494


namespace average_towel_price_l3644_364440

def towel_price_problem (price1 price2 price3 : ℕ) (quantity1 quantity2 quantity3 : ℕ) : Prop :=
  let total_cost := price1 * quantity1 + price2 * quantity2 + price3 * quantity3
  let total_quantity := quantity1 + quantity2 + quantity3
  (total_cost : ℚ) / total_quantity = 205

theorem average_towel_price :
  towel_price_problem 100 150 500 3 5 2 := by
  sorry

end average_towel_price_l3644_364440


namespace distance_between_points_l3644_364483

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (6, 7)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 41 := by
  sorry

end distance_between_points_l3644_364483


namespace rectangle_perimeter_equals_26_l3644_364422

/-- Given a triangle with sides 5, 12, and 13 units, and a rectangle with width 3 units
    and area equal to the triangle's area, the perimeter of the rectangle is 26 units. -/
theorem rectangle_perimeter_equals_26 (triangle_side1 triangle_side2 triangle_side3 : ℝ)
  (rectangle_width : ℝ) (h1 : triangle_side1 = 5)
  (h2 : triangle_side2 = 12) (h3 : triangle_side3 = 13) (h4 : rectangle_width = 3)
  (h5 : (1/2) * triangle_side1 * triangle_side2 = rectangle_width * (((1/2) * triangle_side1 * triangle_side2) / rectangle_width)) :
  2 * (rectangle_width + (((1/2) * triangle_side1 * triangle_side2) / rectangle_width)) = 26 :=
by sorry

end rectangle_perimeter_equals_26_l3644_364422


namespace solution_set_for_specific_values_minimum_value_for_general_case_l3644_364436

-- Define the function f
def f (x a b : ℝ) : ℝ := |x - a| + |x + b|

-- Theorem for part (I)
theorem solution_set_for_specific_values (x : ℝ) :
  let a := 1
  let b := 2
  (f x a b ≤ 5) ↔ (x ∈ Set.Icc (-3) 2) :=
sorry

-- Theorem for part (II)
theorem minimum_value_for_general_case (x a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + 4*b = 2*a*b) :
  f x a b ≥ 9/2 :=
sorry

end solution_set_for_specific_values_minimum_value_for_general_case_l3644_364436


namespace adams_airplane_change_l3644_364455

/-- The change Adam will receive when buying an airplane -/
def adams_change (adams_money : ℚ) (airplane_cost : ℚ) : ℚ :=
  adams_money - airplane_cost

/-- Theorem: Adam's change when buying an airplane -/
theorem adams_airplane_change :
  adams_change 5 4.28 = 0.72 := by sorry

end adams_airplane_change_l3644_364455


namespace characterize_no_solution_set_l3644_364427

/-- The set of real numbers a for which the equation has no solution -/
def NoSolutionSet : Set ℝ :=
  {a | ∀ x, 9 * |x - 4*a| + |x - a^2| + 8*x - 4*a ≠ 0}

/-- The theorem stating the characterization of the set where the equation has no solution -/
theorem characterize_no_solution_set :
  NoSolutionSet = {a | a < -24 ∨ a > 0} :=
by sorry

end characterize_no_solution_set_l3644_364427


namespace prime_arithmetic_seq_large_diff_l3644_364490

/-- A sequence of 15 different positive prime numbers in arithmetic progression -/
structure PrimeArithmeticSequence where
  terms : Fin 15 → ℕ
  is_prime : ∀ i, Nat.Prime (terms i)
  is_arithmetic : ∀ i j k, i.val + k.val = j.val → terms i + terms k = 2 * terms j
  is_distinct : ∀ i j, i ≠ j → terms i ≠ terms j

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : PrimeArithmeticSequence) : ℕ :=
  seq.terms 1 - seq.terms 0

/-- Theorem: The common difference of a sequence of 15 different positive primes
    in arithmetic progression is greater than 30000 -/
theorem prime_arithmetic_seq_large_diff (seq : PrimeArithmeticSequence) :
  common_difference seq > 30000 := by
  sorry

end prime_arithmetic_seq_large_diff_l3644_364490


namespace total_dog_legs_l3644_364415

/-- Proves that the total number of dog legs on a street is 400, given the conditions. -/
theorem total_dog_legs (total_animals : ℕ) (cat_fraction : ℚ) (dog_legs : ℕ) : 
  total_animals = 300 →
  cat_fraction = 2/3 →
  dog_legs = 4 →
  (total_animals * (1 - cat_fraction) : ℚ).num * dog_legs = 400 := by
  sorry

end total_dog_legs_l3644_364415


namespace sale_price_lower_than_original_l3644_364448

theorem sale_price_lower_than_original (x : ℝ) (h : x > 0) : 0.75 * (1.30 * x) < x := by
  sorry

#check sale_price_lower_than_original

end sale_price_lower_than_original_l3644_364448


namespace division_problem_l3644_364480

theorem division_problem : (120 : ℚ) / ((6 : ℚ) / 2 * 3) = 120 / 9 := by
  sorry

end division_problem_l3644_364480


namespace expression_evaluation_l3644_364402

theorem expression_evaluation :
  let x : ℝ := -2
  let y : ℝ := 1
  let z : ℝ := 1
  let w : ℝ := 3
  (x^2 * y^2 * z^2) - (x^2 * y * z^2) + (y / w) * Real.sin (x * z) = -(1/3) * Real.sin 2 :=
by sorry

end expression_evaluation_l3644_364402


namespace cos_theta_range_l3644_364496

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 21 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the center of circle2
def O : ℝ × ℝ := (0, 0)

-- Define a point P on circle1
def P : ℝ × ℝ := sorry

-- Define points A and B where tangents from P touch circle2
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the angle θ between vectors PA and PB
def θ : ℝ := sorry

-- State the theorem
theorem cos_theta_range :
  circle1 P.1 P.2 →
  circle2 A.1 A.2 →
  circle2 B.1 B.2 →
  (1 : ℝ) / 9 ≤ Real.cos θ ∧ Real.cos θ ≤ 41 / 49 :=
sorry

end cos_theta_range_l3644_364496


namespace quadratic_inequality_solution_set_l3644_364492

theorem quadratic_inequality_solution_set (a : ℝ) (h : a > 0) :
  let solution_set := {x : ℝ | a * x^2 - (a + 2) * x + 2 ≥ 0}
  (a = 2 → solution_set = Set.univ) ∧
  (0 < a ∧ a < 2 → solution_set = Set.Iic 1 ∪ Set.Ici (2 / a)) ∧
  (a > 2 → solution_set = Set.Iic (2 / a) ∪ Set.Ici 1) :=
by sorry

end quadratic_inequality_solution_set_l3644_364492


namespace only_36_is_perfect_square_l3644_364488

theorem only_36_is_perfect_square : 
  (∃ n : ℤ, n * n = 36) ∧ 
  (∀ m : ℤ, m * m ≠ 32) ∧ 
  (∀ m : ℤ, m * m ≠ 33) ∧ 
  (∀ m : ℤ, m * m ≠ 34) ∧ 
  (∀ m : ℤ, m * m ≠ 35) :=
by sorry

end only_36_is_perfect_square_l3644_364488


namespace suv_highway_mpg_l3644_364409

/-- The average miles per gallon (mpg) on the highway for an SUV -/
def highway_mpg : ℝ := 12.2

/-- The maximum distance in miles that the SUV can travel on 25 gallons of gasoline -/
def max_distance : ℝ := 305

/-- The amount of gasoline in gallons used to calculate the maximum distance -/
def gasoline_amount : ℝ := 25

theorem suv_highway_mpg :
  highway_mpg = max_distance / gasoline_amount :=
by sorry

end suv_highway_mpg_l3644_364409


namespace quadratic_one_solution_sum_l3644_364441

theorem quadratic_one_solution_sum (a : ℝ) : 
  let f : ℝ → ℝ := λ x => 9*x^2 + a*x + 12*x + 16
  let discriminant := (a + 12)^2 - 4*9*16
  (∃! x, f x = 0) → 
  (∃ a₁ a₂, discriminant = 0 ∧ a = a₁ ∨ a = a₂ ∧ a₁ + a₂ = -24) :=
by sorry

end quadratic_one_solution_sum_l3644_364441


namespace lcm_hcf_problem_l3644_364414

theorem lcm_hcf_problem (a b : ℕ+) 
  (h1 : Nat.lcm a b = 2310)
  (h2 : Nat.gcd a b = 30)
  (h3 : a = 330) : 
  b = 210 := by
  sorry

end lcm_hcf_problem_l3644_364414


namespace paint_coats_calculation_l3644_364475

/-- Proves the number of coats of paint that can be applied given the wall area,
    paint coverage, paint cost, and individual contributions. -/
theorem paint_coats_calculation (wall_area : ℝ) (paint_coverage : ℝ) (paint_cost : ℝ) (contribution : ℝ)
    (h_wall : wall_area = 1600)
    (h_coverage : paint_coverage = 400)
    (h_cost : paint_cost = 45)
    (h_contribution : contribution = 180) :
    ⌊(2 * contribution) / (paint_cost * (wall_area / paint_coverage))⌋ = 2 := by
  sorry

#check paint_coats_calculation

end paint_coats_calculation_l3644_364475


namespace mary_fruit_cost_l3644_364419

/-- The total cost of fruits Mary bought -/
def total_cost (berries apples peaches grapes bananas pineapples : ℚ) : ℚ :=
  berries + apples + peaches + grapes + bananas + pineapples

/-- Theorem stating that the total cost of fruits Mary bought is $52.09 -/
theorem mary_fruit_cost :
  total_cost 11.08 14.33 9.31 7.50 5.25 4.62 = 52.09 := by
  sorry

end mary_fruit_cost_l3644_364419


namespace sin_2x_equals_cos_2x_minus_pi_over_4_l3644_364426

theorem sin_2x_equals_cos_2x_minus_pi_over_4 (x : ℝ) : 
  Real.sin (2 * x) = Real.cos (2 * (x - π / 4)) := by
  sorry

end sin_2x_equals_cos_2x_minus_pi_over_4_l3644_364426


namespace largest_equal_cost_number_equal_cost_3999_largest_equal_cost_is_3999_l3644_364460

/-- Function to calculate the sum of squares of decimal digits -/
def sumSquaresDecimal (n : Nat) : Nat :=
  sorry

/-- Function to calculate the sum of squares of binary digits -/
def sumSquaresBinary (n : Nat) : Nat :=
  sorry

/-- Check if a number has equal costs for both options -/
def hasEqualCosts (n : Nat) : Prop :=
  sumSquaresDecimal n = sumSquaresBinary n

theorem largest_equal_cost_number :
  ∀ n : Nat, n < 5000 → n > 3999 → ¬(hasEqualCosts n) :=
by sorry

theorem equal_cost_3999 : hasEqualCosts 3999 :=
by sorry

theorem largest_equal_cost_is_3999 :
  ∃ n : Nat, n < 5000 ∧ hasEqualCosts n ∧ ∀ m : Nat, m < 5000 → m > n → ¬(hasEqualCosts m) :=
by sorry

end largest_equal_cost_number_equal_cost_3999_largest_equal_cost_is_3999_l3644_364460


namespace same_type_quadratic_root_l3644_364445

theorem same_type_quadratic_root (a : ℝ) : 
  (∃ (k : ℝ), k^2 = 12 ∧ k^2 = 2*a - 5) → a = 4 := by
  sorry

end same_type_quadratic_root_l3644_364445


namespace smallest_solution_quadratic_l3644_364457

theorem smallest_solution_quadratic (x : ℝ) : 
  (3 * x^2 + 36 * x - 60 = x * (x + 17)) → x ≥ -12 :=
by
  sorry

end smallest_solution_quadratic_l3644_364457


namespace greatest_multiple_of_four_under_sqrt_400_l3644_364497

theorem greatest_multiple_of_four_under_sqrt_400 :
  ∀ x : ℕ, 
    x > 0 → 
    (∃ k : ℕ, x = 4 * k) → 
    x^2 < 400 → 
    x ≤ 16 ∧ 
    (∀ y : ℕ, y > 0 → (∃ m : ℕ, y = 4 * m) → y^2 < 400 → y ≤ x) :=
by sorry

end greatest_multiple_of_four_under_sqrt_400_l3644_364497


namespace five_cubes_volume_l3644_364489

/-- The volume of a cube with edge length s -/
def cubeVolume (s : ℝ) : ℝ := s ^ 3

/-- The total volume of n cubes, each with edge length s -/
def totalVolume (n : ℕ) (s : ℝ) : ℝ := n * cubeVolume s

/-- Theorem: The total volume of five cubes with edge length 6 feet is 1080 cubic feet -/
theorem five_cubes_volume : totalVolume 5 6 = 1080 := by
  sorry

end five_cubes_volume_l3644_364489


namespace intersection_of_A_and_B_l3644_364477

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (2 * x - x^2)}

-- Define set B
def B : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 1 2 := by sorry

end intersection_of_A_and_B_l3644_364477


namespace min_value_of_z3_l3644_364473

open Complex

theorem min_value_of_z3 (z₁ z₂ z₃ : ℂ) 
  (h1 : ∃ (a : ℝ), z₁ / z₂ = Complex.I * a)
  (h2 : abs z₁ = 1)
  (h3 : abs z₂ = 1)
  (h4 : abs (z₁ + z₂ + z₃) = 1) :
  abs z₃ ≥ Real.sqrt 2 - 1 := by
sorry

end min_value_of_z3_l3644_364473


namespace count_multiples_of_four_l3644_364458

theorem count_multiples_of_four : ∃ (n : ℕ), n = (Finset.filter (fun x => x % 4 = 0 ∧ x > 300 ∧ x < 700) (Finset.range 700)).card ∧ n = 99 := by
  sorry

end count_multiples_of_four_l3644_364458


namespace unique_solution_characterization_l3644_364485

/-- The set of real numbers a for which the system has a unique solution -/
def UniqueSystemSolutionSet : Set ℝ :=
  {a | a < -5 ∨ a > -1}

/-- The system of equations -/
def SystemEquations (x y a : ℝ) : Prop :=
  x = 4 * Real.sqrt y + a ∧ y^2 - x^2 + 3*y - 5*x - 4 = 0

theorem unique_solution_characterization (a : ℝ) :
  (∃! p : ℝ × ℝ, SystemEquations p.1 p.2 a) ↔ a ∈ UniqueSystemSolutionSet :=
by sorry

end unique_solution_characterization_l3644_364485


namespace intersection_complement_equality_l3644_364465

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3}
def B : Set Nat := {3, 4, 5}

theorem intersection_complement_equality : A ∩ (U \ B) = {1} := by
  sorry

end intersection_complement_equality_l3644_364465


namespace hyperbola_iff_mn_negative_l3644_364456

/-- A hyperbola is represented by the equation x²/m + y²/n = 1 where m and n are real numbers. -/
def IsHyperbola (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / m + y^2 / n = 1

/-- The condition mn < 0 is both necessary and sufficient for the equation x²/m + y²/n = 1 
    to represent a hyperbola. -/
theorem hyperbola_iff_mn_negative (m n : ℝ) : IsHyperbola m n ↔ m * n < 0 := by
  sorry

end hyperbola_iff_mn_negative_l3644_364456


namespace fraction_addition_subtraction_l3644_364449

theorem fraction_addition_subtraction :
  (1 / 4 : ℚ) + (3 / 8 : ℚ) - (1 / 8 : ℚ) = (1 / 2 : ℚ) := by
  sorry

end fraction_addition_subtraction_l3644_364449


namespace minimal_sum_roots_and_qtilde_value_l3644_364474

/-- Represents a quadratic polynomial q(x) = x^2 - (a+b)x + ab -/
def QuadPoly (a b : ℝ) (x : ℝ) : ℝ :=
  x^2 - (a + b) * x + a * b

/-- The condition that q(q(x)) = 0 has exactly three real solutions -/
def HasThreeSolutions (a b : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    QuadPoly a b (QuadPoly a b x) = 0 ∧
    QuadPoly a b (QuadPoly a b y) = 0 ∧
    QuadPoly a b (QuadPoly a b z) = 0 ∧
    ∀ w : ℝ, QuadPoly a b (QuadPoly a b w) = 0 → w = x ∨ w = y ∨ w = z

/-- The sum of roots of q(x) = 0 -/
def SumOfRoots (a b : ℝ) : ℝ := a + b

/-- The polynomial ̃q(x) = x^2 + 2x + 1 -/
def QTilde (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem minimal_sum_roots_and_qtilde_value :
  ∀ a b : ℝ,
  HasThreeSolutions a b →
  (∀ c d : ℝ, HasThreeSolutions c d → SumOfRoots a b ≤ SumOfRoots c d) →
  (QuadPoly a b = QTilde) ∧ QTilde 2 = 9 := by sorry

end minimal_sum_roots_and_qtilde_value_l3644_364474


namespace hyperbola_eccentricity_l3644_364404

/-- Given a hyperbola with the following properties:
  - Point P is on the right branch of the hyperbola (x²/a² - y²/b² = 1), where a > 0 and b > 0
  - F₁ and F₂ are the left and right foci of the hyperbola
  - (OP + OF₂) · F₂P = 0, where O is the origin
  - |PF₁| = √3|PF₂|
  Its eccentricity is √3 + 1 -/
theorem hyperbola_eccentricity (a b : ℝ) (P F₁ F₂ O : ℝ × ℝ) 
  (h_a : a > 0) (h_b : b > 0)
  (h_P : (P.1^2 / a^2) - (P.2^2 / b^2) = 1)
  (h_foci : F₁.1 < 0 ∧ F₂.1 > 0)
  (h_origin : O = (0, 0))
  (h_perpendicular : (P - O + (F₂ - O)) • (P - F₂) = 0)
  (h_distance_ratio : ‖P - F₁‖ = Real.sqrt 3 * ‖P - F₂‖) :
  let c := ‖F₂ - O‖
  c / a = Real.sqrt 3 + 1 := by
  sorry

end hyperbola_eccentricity_l3644_364404


namespace album_difference_l3644_364406

/-- Represents the number of albums each person has -/
structure AlbumCounts where
  adele : ℕ
  bridget : ℕ
  katrina : ℕ
  miriam : ℕ

/-- The conditions of the problem -/
def problem_conditions (counts : AlbumCounts) : Prop :=
  counts.miriam = 5 * counts.katrina ∧
  counts.katrina = 6 * counts.bridget ∧
  counts.bridget < counts.adele ∧
  counts.adele + counts.bridget + counts.katrina + counts.miriam = 585 ∧
  counts.adele = 30

/-- The theorem to be proved -/
theorem album_difference (counts : AlbumCounts) 
  (h : problem_conditions counts) : 
  counts.adele - counts.bridget = 15 := by
  sorry

end album_difference_l3644_364406


namespace number_of_basic_events_l3644_364401

/-- The number of ways to choose 2 items from a set of 3 items -/
def choose_two_from_three : ℕ := 3

/-- The set of interest groups -/
def interest_groups : Finset String := {"Mathematics", "Computer Science", "Model Aviation"}

/-- Xiao Ming must join exactly two groups -/
def join_two_groups (groups : Finset String) : Finset (Finset String) :=
  groups.powerset.filter (fun s => s.card = 2)

theorem number_of_basic_events :
  (join_two_groups interest_groups).card = choose_two_from_three := by sorry

end number_of_basic_events_l3644_364401


namespace namjoon_has_14_pencils_l3644_364499

/-- Represents the number of pencils in a dozen -/
def dozen : ℕ := 12

/-- Represents the number of dozens Taehyung bought -/
def bought_dozens : ℕ := 2

/-- Represents the total number of pencils Taehyung bought -/
def total_pencils : ℕ := bought_dozens * dozen

/-- Represents the number of pencils Taehyung has -/
def taehyung_pencils : ℕ := total_pencils / 2

/-- Represents the number of pencils Namjoon has -/
def namjoon_pencils : ℕ := taehyung_pencils + 4

theorem namjoon_has_14_pencils : namjoon_pencils = 14 := by
  sorry

end namjoon_has_14_pencils_l3644_364499
