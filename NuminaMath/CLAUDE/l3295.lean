import Mathlib

namespace f_satisfies_data_points_l3295_329568

/-- The function that relates x and y --/
def f (x : ℕ) : ℕ := x^2 + x

/-- The set of data points from the table --/
def data_points : List (ℕ × ℕ) := [(1, 2), (2, 6), (3, 12), (4, 20), (5, 30)]

/-- Theorem stating that the function f satisfies all data points --/
theorem f_satisfies_data_points : ∀ (point : ℕ × ℕ), point ∈ data_points → f point.1 = point.2 := by
  sorry

#check f_satisfies_data_points

end f_satisfies_data_points_l3295_329568


namespace grid_bottom_right_value_l3295_329598

/-- Represents a 4x4 grid of rational numbers -/
def Grid := Fin 4 → Fin 4 → ℚ

/-- Checks if a sequence of 4 rational numbers forms an arithmetic progression -/
def isArithmeticSequence (s : Fin 4 → ℚ) : Prop :=
  ∃ d : ℚ, ∀ i : Fin 3, s (i + 1) - s i = d

/-- A grid satisfying the problem conditions -/
def validGrid (g : Grid) : Prop :=
  (∀ i : Fin 4, isArithmeticSequence (λ j ↦ g i j)) ∧  -- Each row is an arithmetic sequence
  (∀ j : Fin 4, isArithmeticSequence (λ i ↦ g i j)) ∧  -- Each column is an arithmetic sequence
  g 0 0 = 1 ∧ g 1 0 = 4 ∧ g 2 0 = 7 ∧ g 3 0 = 10 ∧     -- First column values
  g 2 3 = 25 ∧ g 3 2 = 36                              -- Given values in the grid

theorem grid_bottom_right_value (g : Grid) (h : validGrid g) : g 3 3 = 37 := by
  sorry

end grid_bottom_right_value_l3295_329598


namespace parallel_lines_m_value_l3295_329522

/-- Two lines are parallel if their slopes are equal -/
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

theorem parallel_lines_m_value (m : ℝ) :
  let l₁ : ℝ → ℝ → Prop := λ x y => 2 * x + m * y + 1 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y => y = 3 * x - 1
  parallel (-1/2) (1/3) → m = -2/3 := by sorry

end parallel_lines_m_value_l3295_329522


namespace debt_calculation_l3295_329573

theorem debt_calculation (initial_debt additional_borrowing : ℕ) :
  initial_debt = 20 →
  additional_borrowing = 15 →
  initial_debt + additional_borrowing = 35 :=
by sorry

end debt_calculation_l3295_329573


namespace percent_of_125_l3295_329519

theorem percent_of_125 : ∃ p : ℚ, p * 125 / 100 = 70 ∧ p = 56 := by sorry

end percent_of_125_l3295_329519


namespace min_side_length_l3295_329508

theorem min_side_length (AB AC DC BD BC : ℕ) : 
  AB = 7 → AC = 15 → DC = 10 → BD = 25 → BC > 0 →
  (AB + BC > AC) → (AC + BC > AB) →
  (BD + DC > BC) → (BD + BC > DC) →
  BC ≥ 15 ∧ ∃ (BC : ℕ), BC = 15 ∧ 
    AB + BC > AC ∧ AC + BC > AB ∧
    BD + DC > BC ∧ BD + BC > DC :=
by sorry

end min_side_length_l3295_329508


namespace sallys_shopping_problem_l3295_329542

/-- Sally's shopping problem -/
theorem sallys_shopping_problem 
  (peaches_price_after_coupon : ℝ) 
  (coupon_value : ℝ)
  (total_spent : ℝ)
  (h1 : peaches_price_after_coupon = 12.32)
  (h2 : coupon_value = 3)
  (h3 : total_spent = 23.86) :
  total_spent - (peaches_price_after_coupon + coupon_value) = 8.54 := by
sorry

end sallys_shopping_problem_l3295_329542


namespace product_102_105_l3295_329524

theorem product_102_105 : 102 * 105 = 10710 := by
  sorry

end product_102_105_l3295_329524


namespace quadratic_properties_l3295_329563

-- Define the quadratic equation
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_properties (a b c : ℝ) (h : a ≠ 0) :
  (a - b + c = 0 → discriminant a b c ≥ 0) ∧
  (quadratic_equation a b c 1 ∧ quadratic_equation a b c 2 → 2*a - c = 0) ∧
  ((∃ x y : ℝ, x ≠ y ∧ a * x^2 + c = 0 ∧ a * y^2 + c = 0) →
    ∃ z : ℝ, quadratic_equation a b c z) ∧
  (b = 2*a + c → ∃ x y : ℝ, x ≠ y ∧ quadratic_equation a b c x ∧ quadratic_equation a b c y) :=
sorry

end quadratic_properties_l3295_329563


namespace hybrid_rice_scientific_notation_l3295_329500

theorem hybrid_rice_scientific_notation :
  ∃ n : ℕ, 250000000 = (2.5 : ℝ) * (10 : ℝ) ^ n ∧ n = 8 := by
  sorry

end hybrid_rice_scientific_notation_l3295_329500


namespace original_price_after_percentage_changes_l3295_329505

theorem original_price_after_percentage_changes
  (d r s : ℝ) 
  (h1 : 0 < r ∧ r < 100) 
  (h2 : 0 < s ∧ s < 100) 
  (h3 : s < r) :
  let x := (d * 10000) / (10000 + 100 * (r - s) - r * s)
  x * (1 + r / 100) * (1 - s / 100) = d :=
by sorry

end original_price_after_percentage_changes_l3295_329505


namespace isosceles_triangle_sides_l3295_329515

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  -- The length of the two equal sides
  a : ℝ
  -- The length of the base
  b : ℝ
  -- The height of the triangle
  h : ℝ
  -- The radius of the inscribed circle
  r : ℝ
  -- The perimeter is 56
  perimeter_eq : a + a + b = 56
  -- The radius is 2/7 of the height
  radius_height_ratio : r = 2/7 * h
  -- The height relates to sides by Pythagorean theorem
  height_pythagorean : h^2 + (b/2)^2 = a^2
  -- The area can be calculated using sides and radius
  area_eq : (a + a + b) * r / 2 = b * h / 2

/-- The sides of the isosceles triangle with the given properties are 16, 20, and 20 -/
theorem isosceles_triangle_sides (t : IsoscelesTriangle) : t.a = 20 ∧ t.b = 16 := by
  sorry

end isosceles_triangle_sides_l3295_329515


namespace factor_expression_l3295_329557

theorem factor_expression (b : ℝ) : 145 * b^2 + 29 * b = 29 * b * (5 * b + 1) := by
  sorry

end factor_expression_l3295_329557


namespace right_angle_complementary_angle_l3295_329533

theorem right_angle_complementary_angle (x : ℝ) : 
  x + 23 = 90 → x = 67 := by
  sorry

end right_angle_complementary_angle_l3295_329533


namespace cubic_function_one_zero_l3295_329547

/-- Given a cubic function f(x) = -x^3 - x on the interval [m, n] where f(m) * f(n) < 0,
    f(x) has exactly one zero in the open interval (m, n). -/
theorem cubic_function_one_zero (m n : ℝ) (hm : m < n)
  (f : ℝ → ℝ) (hf : ∀ x, f x = -x^3 - x)
  (h_neg : f m * f n < 0) :
  ∃! x, m < x ∧ x < n ∧ f x = 0 :=
sorry

end cubic_function_one_zero_l3295_329547


namespace multiple_of_x_l3295_329580

theorem multiple_of_x (x y k : ℤ) 
  (eq1 : k * x + y = 34)
  (eq2 : 2 * x - y = 20)
  (y_sq : y^2 = 4) :
  k = 4 := by
sorry

end multiple_of_x_l3295_329580


namespace harkamal_mangoes_purchase_l3295_329590

/-- The amount of mangoes purchased by Harkamal -/
def mangoes : ℕ := sorry

theorem harkamal_mangoes_purchase :
  let grapes_kg : ℕ := 8
  let grapes_rate : ℕ := 70
  let mango_rate : ℕ := 50
  let total_paid : ℕ := 1010
  grapes_kg * grapes_rate + mangoes * mango_rate = total_paid →
  mangoes = 9 := by sorry

end harkamal_mangoes_purchase_l3295_329590


namespace at_least_one_real_root_l3295_329560

theorem at_least_one_real_root (m : ℝ) : 
  ∃ x : ℝ, (x^2 - 5*x + m = 0) ∨ (2*x^2 + x + 6 - m = 0) := by
  sorry

end at_least_one_real_root_l3295_329560


namespace count_integers_squared_between_200_and_400_l3295_329551

theorem count_integers_squared_between_200_and_400 :
  (Finset.filter (fun x => 200 ≤ x^2 ∧ x^2 ≤ 400) (Finset.range 401)).card = 6 := by
  sorry

end count_integers_squared_between_200_and_400_l3295_329551


namespace store_display_cans_l3295_329509

/-- Represents the number of cans in each layer of the display -/
def canSequence : ℕ → ℚ
  | 0 => 30
  | n + 1 => canSequence n - 3

/-- The number of layers in the display -/
def numLayers : ℕ := 11

/-- The total number of cans in the display -/
def totalCans : ℚ := (numLayers : ℚ) * (canSequence 0 + canSequence (numLayers - 1)) / 2

theorem store_display_cans : totalCans = 170.5 := by
  sorry

end store_display_cans_l3295_329509


namespace oranges_for_three_rubles_l3295_329525

/-- Given that 25 oranges cost as many rubles as can be bought for 1 ruble,
    prove that 15 oranges can be bought for 3 rubles -/
theorem oranges_for_three_rubles : ∀ x : ℝ,
  (25 : ℝ) / x = x →  -- 25 oranges cost x rubles, and x oranges can be bought for 1 ruble
  (3 : ℝ) * x = 15 :=  -- 15 oranges can be bought for 3 rubles
by
  sorry

end oranges_for_three_rubles_l3295_329525


namespace profit_percentage_is_40_percent_l3295_329536

/-- The percentage of puppies that can be sold for a greater profit -/
def profitable_puppies_percentage (total_puppies : ℕ) (puppies_with_more_than_4_spots : ℕ) : ℚ :=
  (puppies_with_more_than_4_spots : ℚ) / (total_puppies : ℚ) * 100

/-- Theorem stating that the percentage of puppies that can be sold for a greater profit is 40% -/
theorem profit_percentage_is_40_percent :
  profitable_puppies_percentage 20 8 = 40 := by
  sorry

end profit_percentage_is_40_percent_l3295_329536


namespace sum_of_numbers_l3295_329575

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x * y = 16) (h4 : 1 / x = 3 * (1 / y)) : 
  x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end sum_of_numbers_l3295_329575


namespace division_count_correct_l3295_329523

def num_couples : ℕ := 5
def first_group_size : ℕ := 6
def min_couples_in_first_group : ℕ := 2

/-- The number of ways to divide 5 couples into two groups, 
    where the first group contains 6 people including at least two couples. -/
def num_divisions : ℕ := 130

theorem division_count_correct : 
  ∀ (n : ℕ) (k : ℕ) (m : ℕ),
  n = num_couples → 
  k = first_group_size → 
  m = min_couples_in_first_group →
  num_divisions = (Nat.choose n 2 * (Nat.choose ((n - 2) * 2) 2 - Nat.choose (n - 2) 1)) + 
                   Nat.choose n 3 :=
by sorry

end division_count_correct_l3295_329523


namespace parabola_intersection_distance_l3295_329554

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line passing through the focus with inclination angle π/4
def line (x y : ℝ) : Prop := y = x - 1

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  line A.1 A.2 ∧ line B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem parabola_intersection_distance 
  (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by
  sorry

end parabola_intersection_distance_l3295_329554


namespace recipe_total_cups_l3295_329514

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients used in a recipe -/
def total_cups (ratio : RecipeRatio) (sugar_cups : ℕ) : ℕ :=
  let part_value := sugar_cups / ratio.sugar
  (ratio.butter + ratio.flour + ratio.sugar) * part_value

/-- Theorem: Given a recipe with ratio 2:7:5 and 10 cups of sugar, the total is 28 cups -/
theorem recipe_total_cups :
  let ratio := RecipeRatio.mk 2 7 5
  total_cups ratio 10 = 28 := by
  sorry

end recipe_total_cups_l3295_329514


namespace hundredth_term_equals_30503_l3295_329553

/-- A sequence of geometric designs -/
def f (n : ℕ) : ℕ := 3 * n^2 + 5 * n + 3

/-- The theorem stating that the 100th term of the sequence equals 30503 -/
theorem hundredth_term_equals_30503 :
  f 0 = 3 ∧ f 1 = 11 ∧ f 2 = 25 ∧ f 3 = 45 → f 100 = 30503 := by
  sorry

end hundredth_term_equals_30503_l3295_329553


namespace difference_of_two_numbers_l3295_329565

theorem difference_of_two_numbers (x y : ℝ) 
  (sum_eq : x + y = 30) 
  (product_eq : x * y = 140) : 
  |x - y| = Real.sqrt 340 := by sorry

end difference_of_two_numbers_l3295_329565


namespace antonios_meatballs_l3295_329595

/-- Given the conditions of Antonio's meatball preparation, prove the amount of hamburger per meatball. -/
theorem antonios_meatballs (family_members : ℕ) (total_hamburger : ℝ) (antonios_meatballs : ℕ) :
  family_members = 8 →
  total_hamburger = 4 →
  antonios_meatballs = 4 →
  (total_hamburger / (family_members * antonios_meatballs) : ℝ) = 0.125 := by
  sorry

#check antonios_meatballs

end antonios_meatballs_l3295_329595


namespace hou_debang_developed_alkali_process_l3295_329511

-- Define a type for scientists
inductive Scientist
| HouDebang
| HouGuangtian
| HouXianglin
| HouXueyu

-- Define a type for chemical processes
structure ChemicalProcess where
  name : String
  developer : Scientist
  developmentDate : String

-- Define the Hou's Alkali Process
def housAlkaliProcess : ChemicalProcess := {
  name := "Hou's Alkali Process",
  developer := Scientist.HouDebang,
  developmentDate := "March 1941"
}

-- Theorem statement
theorem hou_debang_developed_alkali_process :
  housAlkaliProcess.developer = Scientist.HouDebang ∧
  housAlkaliProcess.name = "Hou's Alkali Process" ∧
  housAlkaliProcess.developmentDate = "March 1941" :=
by sorry

end hou_debang_developed_alkali_process_l3295_329511


namespace g_50_equals_zero_l3295_329501

-- Define φ(n) as the number of positive integers not exceeding n that are coprime to n
def phi (n : ℕ) : ℕ := sorry

-- Define g(n) that satisfies the given condition
def g (n : ℕ) : ℤ := sorry

-- Define the sum of g(d) over all positive divisors d of n
def sum_g_divisors (n : ℕ) : ℤ := sorry

-- State the condition that g(n) satisfies
axiom g_condition (n : ℕ) : sum_g_divisors n = phi n

-- Theorem to prove
theorem g_50_equals_zero : g 50 = 0 := by sorry

end g_50_equals_zero_l3295_329501


namespace hot_dogs_remainder_l3295_329599

theorem hot_dogs_remainder : 34582918 % 6 = 4 := by
  sorry

end hot_dogs_remainder_l3295_329599


namespace geometric_sequence_a5_l3295_329562

/-- A geometric sequence with real terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Theorem: In a geometric sequence with a₁ = 1 and a₃ = 2, a₅ = 4 -/
theorem geometric_sequence_a5 (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 1) 
  (h_a3 : a 3 = 2) : 
  a 5 = 4 := by
sorry

end geometric_sequence_a5_l3295_329562


namespace increasing_function_m_range_l3295_329526

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + m

theorem increasing_function_m_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, -2 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) →
  m ∈ Set.Ici 4 :=
sorry

end increasing_function_m_range_l3295_329526


namespace product_xy_is_eight_l3295_329521

theorem product_xy_is_eight (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 64)
  (h2 : (16:ℝ)^(x+y) / (4:ℝ)^(4*y) = 256) : 
  x * y = 8 := by
sorry

end product_xy_is_eight_l3295_329521


namespace parallel_line_through_point_l3295_329529

-- Define the type for points in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the type for lines in 2D space
structure Line2D where
  f : ℝ → ℝ → ℝ

-- Define the property of a point being on a line
def PointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.f p.x p.y = 0

-- Define the property of two lines being parallel
def ParallelLines (l1 l2 : Line2D) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), l1.f x y = k * l2.f x y

theorem parallel_line_through_point
  (l : Line2D) (p1 p2 : Point2D)
  (h1 : PointOnLine p1 l)
  (h2 : ¬PointOnLine p2 l) :
  let l2 : Line2D := { f := λ x y => l.f x y - l.f p2.x p2.y }
  ParallelLines l l2 ∧ PointOnLine p2 l2 :=
by
  sorry


end parallel_line_through_point_l3295_329529


namespace cake_icing_theorem_l3295_329507

/-- Represents a rectangular prism cake with icing on specific sides -/
structure CakeWithIcing where
  length : ℕ
  width : ℕ
  height : ℕ
  hasTopIcing : Bool
  hasFrontIcing : Bool
  hasBackIcing : Bool

/-- Counts the number of 1x1x1 cubes with icing on exactly two sides -/
def countCubesWithTwoSidesIced (cake : CakeWithIcing) : ℕ :=
  sorry

/-- The main theorem stating that a 5x5x3 cake with top, front, and back icing
    will have exactly 30 small cubes with icing on two sides when divided into 1x1x1 cubes -/
theorem cake_icing_theorem :
  let cake : CakeWithIcing := {
    length := 5,
    width := 5,
    height := 3,
    hasTopIcing := true,
    hasFrontIcing := true,
    hasBackIcing := true
  }
  countCubesWithTwoSidesIced cake = 30 := by
  sorry

end cake_icing_theorem_l3295_329507


namespace correct_average_l3295_329528

theorem correct_average (n : ℕ) (initial_avg : ℚ) (increase : ℚ) : 
  n = 10 →
  initial_avg = 5 →
  increase = 10 →
  (n : ℚ) * initial_avg + increase = n * 6 := by
  sorry

end correct_average_l3295_329528


namespace simplify_and_evaluate_expression_l3295_329506

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = 3) :
  (1 - (a - 2) / (a^2 - 4)) / ((a^2 + a) / (a^2 + 4*a + 4)) = 5/3 := by
  sorry

end simplify_and_evaluate_expression_l3295_329506


namespace race_length_correct_l3295_329583

/-- Represents the race scenario -/
structure Race where
  length : ℝ
  samTime : ℝ
  johnTime : ℝ
  headStart : ℝ

/-- The given race conditions -/
def givenRace : Race where
  length := 126
  samTime := 13
  johnTime := 18
  headStart := 35

/-- Theorem stating that the given race length satisfies all conditions -/
theorem race_length_correct (r : Race) : 
  r.samTime = 13 ∧ 
  r.johnTime = r.samTime + 5 ∧ 
  r.headStart = 35 ∧
  r.length / r.samTime * r.samTime = r.length / r.johnTime * r.samTime + r.headStart →
  r.length = 126 := by
  sorry

#check race_length_correct givenRace

end race_length_correct_l3295_329583


namespace battery_current_l3295_329530

/-- Given a battery with voltage 48V, prove that when connected to a 12Ω resistance, 
    the resulting current is 4A. -/
theorem battery_current (V R I : ℝ) : 
  V = 48 → R = 12 → I = V / R → I = 4 := by sorry

end battery_current_l3295_329530


namespace hyperbola_eccentricity_sqrt_two_l3295_329534

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a parabola with parameter p -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- Theorem stating that under given conditions, the eccentricity of the hyperbola is √2 -/
theorem hyperbola_eccentricity_sqrt_two (h : Hyperbola) (p : Parabola)
  (h_focus : h.a * h.a - h.b * h.b = p.p * h.a) -- Right focus of hyperbola coincides with focus of parabola
  (h_intersection : ∃ A B : ℝ × ℝ, 
    A.1^2 / h.a^2 - A.2^2 / h.b^2 = 1 ∧
    B.1^2 / h.a^2 - B.2^2 / h.b^2 = 1 ∧
    A.1 = -p.p/2 ∧ B.1 = -p.p/2) -- Directrix of parabola intersects hyperbola at A and B
  (h_asymptotes : ∃ C D : ℝ × ℝ,
    C.2 = h.b / h.a * C.1 ∧
    D.2 = -h.b / h.a * D.1) -- Asymptotes of hyperbola intersect at C and D
  (h_distance : ∃ A B C D : ℝ × ℝ,
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = 2 * ((A.1 - B.1)^2 + (A.2 - B.2)^2)) -- |CD| = √2|AB|
  : Real.sqrt ((h.a^2 - h.b^2) / h.a^2) = Real.sqrt 2 := by
  sorry

end hyperbola_eccentricity_sqrt_two_l3295_329534


namespace union_with_complement_l3295_329593

theorem union_with_complement (I A B : Set ℕ) : 
  I = {1, 2, 3, 4} →
  A = {1} →
  B = {2, 4} →
  A ∪ (I \ B) = {1, 3} := by
  sorry

end union_with_complement_l3295_329593


namespace sqrt_sum_equals_nine_l3295_329518

theorem sqrt_sum_equals_nine :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) - 1 = 9 := by
  sorry

end sqrt_sum_equals_nine_l3295_329518


namespace simplify_radical_product_l3295_329527

theorem simplify_radical_product (x : ℝ) (h : x > 0) :
  2 * Real.sqrt (50 * x^3) * Real.sqrt (45 * x^5) * Real.sqrt (98 * x^7) = 420 * x^7 * Real.sqrt (5 * x) := by
  sorry

end simplify_radical_product_l3295_329527


namespace sum_of_s_and_t_l3295_329537

theorem sum_of_s_and_t (s t : ℕ+) (h : s * (s - t) = 29) : s + t = 57 := by
  sorry

end sum_of_s_and_t_l3295_329537


namespace minimize_sum_squared_distances_l3295_329587

-- Define the points A, B, C
def A : ℝ × ℝ := (3, -1)
def B : ℝ × ℝ := (-1, 4)
def C : ℝ × ℝ := (1, -6)

-- Define the function to calculate the sum of squared distances
def sumSquaredDistances (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  (x - A.1)^2 + (y - A.2)^2 +
  (x - B.1)^2 + (y - B.2)^2 +
  (x - C.1)^2 + (y - C.2)^2

-- Define the point P
def P : ℝ × ℝ := (1, -1)

-- Theorem statement
theorem minimize_sum_squared_distances :
  ∀ Q : ℝ × ℝ, sumSquaredDistances P ≤ sumSquaredDistances Q :=
sorry

end minimize_sum_squared_distances_l3295_329587


namespace soup_bins_calculation_l3295_329570

/-- Given a canned food drive with different types of food, calculate the number of bins of soup. -/
theorem soup_bins_calculation (total_bins vegetables_bins pasta_bins : ℝ) 
  (h1 : vegetables_bins = 0.125)
  (h2 : pasta_bins = 0.5)
  (h3 : total_bins = 0.75) :
  total_bins - (vegetables_bins + pasta_bins) = 0.125 := by
  sorry

#check soup_bins_calculation

end soup_bins_calculation_l3295_329570


namespace least_sum_of_bases_l3295_329503

/-- Represents a number in a given base -/
def NumberInBase (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => acc * base + digit) 0

theorem least_sum_of_bases :
  ∃ (c d : Nat),
    c > 0 ∧ d > 0 ∧
    NumberInBase [5, 8] c = NumberInBase [8, 5] d ∧
    c + d = 15 ∧
    ∀ (c' d' : Nat), c' > 0 → d' > 0 → NumberInBase [5, 8] c' = NumberInBase [8, 5] d' → c' + d' ≥ 15 :=
by sorry

end least_sum_of_bases_l3295_329503


namespace square_side_length_l3295_329576

theorem square_side_length (rectangle_side1 rectangle_side2 : ℝ) 
  (h1 : rectangle_side1 = 9)
  (h2 : rectangle_side2 = 16) :
  ∃ (square_side : ℝ), 
    square_side * square_side = rectangle_side1 * rectangle_side2 ∧ 
    square_side = 12 := by
  sorry

end square_side_length_l3295_329576


namespace range_of_a_for_no_real_roots_l3295_329539

theorem range_of_a_for_no_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a + 1) * x + 1 > 0) ↔ a ∈ Set.Ioo (-3 : ℝ) 1 :=
by sorry

end range_of_a_for_no_real_roots_l3295_329539


namespace simplify_expression_l3295_329588

theorem simplify_expression (a b c : ℝ) : a - (a - b + c) = b - c := by
  sorry

end simplify_expression_l3295_329588


namespace valid_palindrome_count_l3295_329582

def valid_digits : Finset Nat := {0, 7, 8, 9}

def is_palindrome (n : Nat) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def count_valid_palindromes : Nat :=
  (valid_digits.filter (· ≠ 0)).card *
  valid_digits.card ^ 2 *
  valid_digits.card ^ 2 *
  valid_digits.card

theorem valid_palindrome_count :
  count_valid_palindromes = 3072 := by sorry

end valid_palindrome_count_l3295_329582


namespace collinear_points_k_value_l3295_329591

/-- Three points (x1, y1), (x2, y2), and (x3, y3) are collinear if and only if
    the slope between any two pairs of points is equal. -/
def collinear (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

/-- The theorem states that for three collinear points (1, 2), (3, k), and (10, 5),
    the value of k must be 8/3. -/
theorem collinear_points_k_value :
  collinear 1 2 3 k 10 5 → k = 8/3 :=
by
  sorry

end collinear_points_k_value_l3295_329591


namespace f_range_l3295_329531

-- Define the function f
def f (x : ℝ) : ℝ := -(x - 5)^2 + 1

-- Define the domain
def domain : Set ℝ := {x | 2 < x ∧ x < 6}

-- Define the range
def range : Set ℝ := {y | -8 < y ∧ y ≤ 1}

-- Theorem statement
theorem f_range : 
  ∀ y ∈ range, ∃ x ∈ domain, f x = y ∧
  ∀ x ∈ domain, f x ∈ range :=
sorry

end f_range_l3295_329531


namespace min_value_theorem_l3295_329572

theorem min_value_theorem (x : ℝ) (h : x > 4) :
  (x + 5) / Real.sqrt (x - 4) ≥ 6 ∧ ∃ y : ℝ, y > 4 ∧ (y + 5) / Real.sqrt (y - 4) = 6 := by
  sorry

end min_value_theorem_l3295_329572


namespace xiaohong_total_score_l3295_329520

/-- Calculates the total score based on midterm and final exam scores -/
def total_score (midterm_weight : ℝ) (final_weight : ℝ) (midterm_score : ℝ) (final_score : ℝ) : ℝ :=
  midterm_weight * midterm_score + final_weight * final_score

theorem xiaohong_total_score :
  let midterm_weight : ℝ := 0.4
  let final_weight : ℝ := 0.6
  let midterm_score : ℝ := 80
  let final_score : ℝ := 90
  total_score midterm_weight final_weight midterm_score final_score = 86 := by
  sorry

#eval total_score 0.4 0.6 80 90

end xiaohong_total_score_l3295_329520


namespace odd_cube_plus_multiple_l3295_329581

theorem odd_cube_plus_multiple (p m : ℤ) (hp : Odd p) :
  Odd (p^3 + m*p) ↔ Even m :=
sorry

end odd_cube_plus_multiple_l3295_329581


namespace average_expenditure_feb_to_jul_l3295_329561

def average_expenditure_jan_to_jun : ℚ := 4200
def january_expenditure : ℚ := 1200
def july_expenditure : ℚ := 1500

theorem average_expenditure_feb_to_jul :
  let total_jan_to_jun := 6 * average_expenditure_jan_to_jun
  let total_feb_to_jun := total_jan_to_jun - january_expenditure
  let total_feb_to_jul := total_feb_to_jun + july_expenditure
  total_feb_to_jul / 6 = 4250 := by sorry

end average_expenditure_feb_to_jul_l3295_329561


namespace sin_negative_45_degrees_l3295_329550

theorem sin_negative_45_degrees :
  Real.sin ((-45 : ℝ) * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_negative_45_degrees_l3295_329550


namespace kaleb_toy_purchase_l3295_329594

def max_toys_purchasable (initial_savings : ℕ) (allowance : ℕ) (toy_cost : ℕ) : ℕ :=
  (initial_savings + allowance) / toy_cost

theorem kaleb_toy_purchase :
  max_toys_purchasable 21 15 6 = 6 := by
  sorry

end kaleb_toy_purchase_l3295_329594


namespace truck_distance_before_meeting_l3295_329552

/-- The distance between two trucks one minute before they meet, given their initial separation and speeds -/
theorem truck_distance_before_meeting
  (initial_distance : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ)
  (h1 : initial_distance = 4)
  (h2 : speed_A = 45)
  (h3 : speed_B = 60)
  : ∃ (d : ℝ), d = 250 / 1000 ∧ d = initial_distance + speed_A * (1 / 60) - speed_B * (1 / 60) :=
by sorry

end truck_distance_before_meeting_l3295_329552


namespace kekai_money_left_l3295_329532

def garage_sale_problem (num_shirts num_pants : ℕ) (price_shirt price_pants : ℚ) : ℚ :=
  let total_earned := num_shirts * price_shirt + num_pants * price_pants
  let amount_to_parents := total_earned / 2
  total_earned - amount_to_parents

theorem kekai_money_left :
  garage_sale_problem 5 5 1 3 = 10 := by
  sorry

end kekai_money_left_l3295_329532


namespace catch_second_messenger_first_is_optimal_l3295_329589

/-- Represents a person's position and movement --/
structure Person where
  position : ℝ
  speed : ℝ
  startTime : ℝ

/-- Represents the problem setup --/
structure ProblemSetup where
  messenger1 : Person
  messenger2 : Person
  cyclist : Person
  startPoint : ℝ

/-- Calculates the time needed for the cyclist to complete the task --/
def timeToComplete (setup : ProblemSetup) (catchSecondFirst : Bool) : ℝ :=
  sorry

/-- Theorem stating that catching the second messenger first is optimal --/
theorem catch_second_messenger_first_is_optimal (setup : ProblemSetup) :
  setup.messenger1.startTime + 0.25 = setup.messenger2.startTime →
  setup.messenger1.speed = setup.messenger2.speed →
  setup.cyclist.speed > setup.messenger1.speed →
  setup.messenger1.position < setup.startPoint →
  setup.messenger2.position > setup.startPoint →
  timeToComplete setup true ≤ timeToComplete setup false :=
sorry

end catch_second_messenger_first_is_optimal_l3295_329589


namespace inequality_property_l3295_329558

theorem inequality_property (a b : ℝ) (h : a < 0 ∧ 0 < b) : 1 / a < 1 / b := by
  sorry

end inequality_property_l3295_329558


namespace exists_x_y_for_3k_l3295_329579

theorem exists_x_y_for_3k (k : ℕ+) : 
  ∃ (x y : ℤ), (¬ 3 ∣ x) ∧ (¬ 3 ∣ y) ∧ (x^2 + 2*y^2 = 3^(k.val)) := by
  sorry

end exists_x_y_for_3k_l3295_329579


namespace manuscript_typing_cost_l3295_329567

/-- The total cost of typing a manuscript with given revision rates and page counts. -/
theorem manuscript_typing_cost
  (initial_rate : ℕ)
  (revision_rate : ℕ)
  (total_pages : ℕ)
  (once_revised_pages : ℕ)
  (twice_revised_pages : ℕ)
  (h1 : initial_rate = 5)
  (h2 : revision_rate = 3)
  (h3 : total_pages = 100)
  (h4 : once_revised_pages = 30)
  (h5 : twice_revised_pages = 20) :
  initial_rate * total_pages +
  revision_rate * once_revised_pages +
  2 * revision_rate * twice_revised_pages = 710 := by
sorry


end manuscript_typing_cost_l3295_329567


namespace no_negative_exponents_l3295_329538

theorem no_negative_exponents (a b c d : Int) 
  (h1 : (5 : ℝ)^a + (5 : ℝ)^b = (3 : ℝ)^c + (3 : ℝ)^d)
  (h2 : Even a) (h3 : Even b) (h4 : Even c) (h5 : Even d) :
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 :=
by sorry

end no_negative_exponents_l3295_329538


namespace problem_parallelogram_area_l3295_329585

/-- A parallelogram in 2D space defined by four vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- The specific parallelogram from the problem -/
def problem_parallelogram : Parallelogram :=
  { v1 := (0, 0)
    v2 := (4, 0)
    v3 := (1, 5)
    v4 := (5, 5) }

/-- Theorem stating that the area of the problem parallelogram is 20 -/
theorem problem_parallelogram_area :
  area problem_parallelogram = 20 := by sorry

end problem_parallelogram_area_l3295_329585


namespace min_value_and_range_l3295_329546

variable (x y z : ℝ)

def t (x y z : ℝ) : ℝ := x^2 + y^2 + 2*z^2

theorem min_value_and_range :
  (x + y + 2*z = 1) →
  (∃ (min : ℝ), ∀ x y z, t x y z ≥ min ∧ ∃ x y z, t x y z = min) ∧
  (t x y z = 1/2 → 0 ≤ z ∧ z ≤ 1/2) :=
by sorry

end min_value_and_range_l3295_329546


namespace symmetric_points_on_parabola_l3295_329513

/-- Given two points on a parabola that are symmetric about a line, prove the value of m -/
theorem symmetric_points_on_parabola (x₁ x₂ y₁ y₂ m : ℝ) : 
  y₁ = 2 * x₁^2 →                   -- A is on the parabola
  y₂ = 2 * x₂^2 →                   -- B is on the parabola
  (y₁ + y₂) / 2 = (x₁ + x₂) / 2 + m →  -- Midpoint of A and B is on y = x + m
  (y₂ - y₁) / (x₂ - x₁) = -1 →      -- Slope of AB is perpendicular to y = x + m
  x₁ * x₂ = -1/2 →                  -- Given condition
  m = 3/2 := by
sorry

end symmetric_points_on_parabola_l3295_329513


namespace sum_of_repeating_decimals_l3295_329504

def repeating_decimal_12 : ℚ := 4 / 33
def repeating_decimal_34 : ℚ := 34 / 99

theorem sum_of_repeating_decimals :
  repeating_decimal_12 + repeating_decimal_34 = 46 / 99 := by
  sorry

end sum_of_repeating_decimals_l3295_329504


namespace f_5_equals_neg_2_l3295_329566

-- Define the inverse function f⁻¹
def f_inv (x : ℝ) : ℝ := 1 + x^2

-- State the theorem
theorem f_5_equals_neg_2 (f : ℝ → ℝ) (h1 : ∀ x < 0, f_inv (f x) = x) (h2 : ∀ y, y < 0 → f (f_inv y) = y) : 
  f 5 = -2 := by
  sorry

end f_5_equals_neg_2_l3295_329566


namespace largest_prime_factor_of_1001_l3295_329597

theorem largest_prime_factor_of_1001 : ∃ p : ℕ, p.Prime ∧ p ∣ 1001 ∧ ∀ q : ℕ, q.Prime → q ∣ 1001 → q ≤ p :=
  sorry

end largest_prime_factor_of_1001_l3295_329597


namespace line_through_points_l3295_329592

/-- A line passing through three given points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- The y-coordinate of a point on the line given its x-coordinate -/
def y_coord (l : Line) (x : ℝ) : ℝ :=
  sorry

theorem line_through_points (l : Line) : 
  l.point1 = (2, 8) ∧ l.point2 = (4, 14) ∧ l.point3 = (6, 20) → 
  y_coord l 50 = 152 := by
  sorry

end line_through_points_l3295_329592


namespace population_after_10_years_l3295_329586

/-- Given an initial population and growth rate, calculates the population after n years -/
def population (M : ℝ) (p : ℝ) (n : ℕ) : ℝ := M * (1 + p) ^ n

/-- Theorem: The population after 10 years with initial population M and growth rate p is M(1+p)^10 -/
theorem population_after_10_years (M p : ℝ) : 
  population M p 10 = M * (1 + p)^10 := by
  sorry

end population_after_10_years_l3295_329586


namespace player_two_wins_l3295_329577

/-- Number of contacts in the microcircuit -/
def num_contacts : ℕ := 2000

/-- Total number of wires initially -/
def total_wires : ℕ := num_contacts * (num_contacts - 1) / 2

/-- Represents a player in the game -/
inductive Player
| One
| Two

/-- Represents the state of the game -/
structure GameState where
  remaining_wires : ℕ
  current_player : Player

/-- Defines a valid move in the game -/
def valid_move (state : GameState) (wires_cut : ℕ) : Prop :=
  match state.current_player with
  | Player.One => wires_cut = 1
  | Player.Two => wires_cut = 2 ∨ wires_cut = 3

/-- Defines the winning condition -/
def is_winning_state (state : GameState) : Prop :=
  state.remaining_wires = 0

/-- Theorem stating that Player Two has a winning strategy -/
theorem player_two_wins : 
  ∃ (strategy : GameState → ℕ), 
    ∀ (game : GameState), 
      game.remaining_wires > 0 → 
      game.current_player = Player.Two → 
      valid_move game (strategy game) ∧ 
      (∀ (opponent_move : ℕ), 
        valid_move (GameState.mk (game.remaining_wires - strategy game) Player.One) opponent_move → 
        is_winning_state (GameState.mk (game.remaining_wires - strategy game - opponent_move) Player.Two)) :=
sorry

end player_two_wins_l3295_329577


namespace four_digit_number_with_specific_remainders_l3295_329516

theorem four_digit_number_with_specific_remainders :
  ∃ n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧
    n % 7 = 3 ∧
    n % 10 = 6 ∧
    n % 12 = 8 ∧
    n % 13 = 2 := by
  sorry

end four_digit_number_with_specific_remainders_l3295_329516


namespace buckingham_palace_visitors_l3295_329548

theorem buckingham_palace_visitors (current_day_visitors previous_day_visitors : ℕ) 
  (h1 : current_day_visitors = 661) 
  (h2 : previous_day_visitors = 600) : 
  current_day_visitors - previous_day_visitors = 61 := by
  sorry

end buckingham_palace_visitors_l3295_329548


namespace term_2020_2187_position_l3295_329549

/-- Sequence term type -/
structure SeqTerm where
  k : Nat
  n : Nat
  h : k % 3 ≠ 0 ∧ 2 ≤ k ∧ k < 3^n

/-- Position of a term in the sequence -/
def termPosition (t : SeqTerm) : Nat :=
  t.k - (t.k / 3)

/-- The main theorem -/
theorem term_2020_2187_position :
  ∃ t : SeqTerm, t.k = 2020 ∧ t.n = 7 ∧ termPosition t = 1347 := by
  sorry

end term_2020_2187_position_l3295_329549


namespace circle_area_l3295_329556

theorem circle_area (d : ℝ) (A : ℝ) (π : ℝ) (h1 : d = 10) (h2 : π = Real.pi) :
  A = π * 25 → A = (π * d^2) / 4 :=
by
  sorry

end circle_area_l3295_329556


namespace fifth_power_sum_equality_l3295_329564

theorem fifth_power_sum_equality : ∃! m : ℕ+, m.val ^ 5 = 144 ^ 5 + 91 ^ 5 + 56 ^ 5 + 19 ^ 5 := by
  sorry

end fifth_power_sum_equality_l3295_329564


namespace line_equation_and_x_intercept_l3295_329502

-- Define the points A and B
def A : ℝ × ℝ := (-2, -3)
def B : ℝ × ℝ := (3, 0)

-- Define the line l
def l (x y : ℝ) : Prop := 5 * x + 3 * y + 2 = 0

-- Define symmetry about a line
def symmetric_about_line (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (M : ℝ × ℝ), l M.1 M.2 ∧ 
  (M.1 = (A.1 + B.1) / 2) ∧ 
  (M.2 = (A.2 + B.2) / 2)

-- Theorem statement
theorem line_equation_and_x_intercept :
  symmetric_about_line A B l →
  (∀ x y, l x y ↔ 5 * x + 3 * y + 2 = 0) ∧
  (∃ x, l x 0 ∧ x = -2/5) :=
sorry

end line_equation_and_x_intercept_l3295_329502


namespace inequality_range_theorem_l3295_329578

/-- The range of values for the real number a that satisfies the inequality
    2*ln(x) ≥ -x^2 + ax - 3 for all x ∈ (0, +∞) is (-∞, 4]. -/
theorem inequality_range_theorem (a : ℝ) : 
  (∀ x > 0, 2 * Real.log x ≥ -x^2 + a*x - 3) ↔ a ≤ 4 :=
by sorry

end inequality_range_theorem_l3295_329578


namespace probability_exact_scenario_verify_conditions_l3295_329574

/-- The probability of drawing all red balls by the 4th draw in a specific scenario -/
def probability_all_red_by_fourth_draw (total_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ) : ℚ :=
  let total_ways := (total_balls.choose 4)
  let favorable_ways := (red_balls.choose 2) * (white_balls.choose 2)
  (favorable_ways : ℚ) / total_ways

/-- The main theorem stating the probability for the given scenario -/
theorem probability_exact_scenario : 
  probability_all_red_by_fourth_draw 10 8 2 = 4338 / 91125 := by
  sorry

/-- Verifies that the conditions of the problem are met -/
theorem verify_conditions : 
  10 = 8 + 2 ∧ 
  8 ≥ 0 ∧ 
  2 ≥ 0 ∧
  10 > 0 := by
  sorry

end probability_exact_scenario_verify_conditions_l3295_329574


namespace a_minus_b_values_l3295_329569

theorem a_minus_b_values (a b : ℤ) (ha : |a| = 7) (hb : |b| = 5) (hab : a < b) :
  a - b = -12 ∨ a - b = -2 :=
sorry

end a_minus_b_values_l3295_329569


namespace points_coplanar_iff_b_eq_neg_one_l3295_329535

/-- Given four points in 3D space, prove they are coplanar iff b = -1 --/
theorem points_coplanar_iff_b_eq_neg_one (b : ℝ) :
  let p1 := (0 : ℝ × ℝ × ℝ)
  let p2 := (1, b, 0)
  let p3 := (0, 1, b^2)
  let p4 := (b^2, 0, 1)
  (∃ (a b c d : ℝ), a • p1 + b • p2 + c • p3 + d • p4 = 0 ∧ (a, b, c, d) ≠ 0) ↔ b = -1 := by
  sorry

#check points_coplanar_iff_b_eq_neg_one

end points_coplanar_iff_b_eq_neg_one_l3295_329535


namespace train_length_l3295_329555

theorem train_length (tree_time : ℝ) (platform_time : ℝ) (platform_length : ℝ) :
  tree_time = 120 →
  platform_time = 220 →
  platform_length = 1000 →
  let train_length := (platform_time * platform_length) / (platform_time - tree_time)
  train_length = 1200 := by
  sorry

#check train_length

end train_length_l3295_329555


namespace newspaper_delivery_patterns_l3295_329510

/-- Represents the number of valid newspaper delivery patterns for n houses -/
def D : ℕ → ℕ
| 0 => 1  -- Base case, one way to deliver to zero houses
| 1 => 2  -- Two ways to deliver to one house (deliver or not)
| 2 => 4  -- Four ways to deliver to two houses
| n + 3 => D (n + 2) + D (n + 1) + D n  -- Recurrence relation

/-- The condition that the last house must receive a newspaper -/
def lastHouseDelivery (n : ℕ) : ℕ := D (n - 1)

/-- The number of houses on the lane -/
def numHouses : ℕ := 12

/-- The theorem stating the number of valid delivery patterns for 12 houses -/
theorem newspaper_delivery_patterns :
  lastHouseDelivery numHouses = 927 := by sorry

end newspaper_delivery_patterns_l3295_329510


namespace selling_price_calculation_l3295_329596

/-- The selling price that yields a 4% higher gain than selling at 340, given a cost of 250 -/
def higher_selling_price (cost : ℝ) (lower_price : ℝ) : ℝ :=
  let lower_gain := lower_price - cost
  let higher_gain := lower_gain * 1.04
  cost + higher_gain

theorem selling_price_calculation :
  higher_selling_price 250 340 = 343.6 := by
  sorry

end selling_price_calculation_l3295_329596


namespace solve_system_l3295_329559

theorem solve_system (x y : ℤ) 
  (h1 : x + y = 260) 
  (h2 : x - y = 200) : 
  y = 30 := by
sorry

end solve_system_l3295_329559


namespace tan_equality_solution_l3295_329545

theorem tan_equality_solution (n : ℤ) :
  -150 < n ∧ n < 150 ∧ Real.tan (n * π / 180) = Real.tan (1340 * π / 180) →
  n = 80 ∨ n = -100 := by
sorry

end tan_equality_solution_l3295_329545


namespace non_shaded_perimeter_theorem_l3295_329584

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem about the perimeter of a non-shaded region in a composite figure -/
theorem non_shaded_perimeter_theorem 
  (large_rect : Rectangle)
  (small_rect : Rectangle)
  (shaded_area : ℝ)
  (h1 : large_rect.length = 12)
  (h2 : large_rect.width = 10)
  (h3 : small_rect.length = 4)
  (h4 : small_rect.width = 3)
  (h5 : shaded_area = 104) : 
  ∃ (non_shaded_rect : Rectangle), 
    abs (perimeter non_shaded_rect - 21.34) < 0.01 := by
  sorry

end non_shaded_perimeter_theorem_l3295_329584


namespace total_cantelopes_l3295_329544

theorem total_cantelopes (fred_cantelopes tim_cantelopes : ℕ) 
  (h1 : fred_cantelopes = 38) 
  (h2 : tim_cantelopes = 44) : 
  fred_cantelopes + tim_cantelopes = 82 := by
sorry

end total_cantelopes_l3295_329544


namespace min_sum_squares_l3295_329512

theorem min_sum_squares (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → x^2 + y^2 ≥ m := by
  sorry

end min_sum_squares_l3295_329512


namespace alfonso_solution_l3295_329540

/-- Alfonso's weekly earnings and financial goals --/
def alfonso_problem (weekday_earnings : ℕ) (weekend_earnings : ℕ) 
  (total_cost : ℕ) (current_savings : ℕ) (desired_remaining : ℕ) : Prop :=
  let weekly_earnings := 5 * weekday_earnings + 2 * weekend_earnings
  let total_needed := total_cost - current_savings + desired_remaining
  let weeks_needed := (total_needed + weekly_earnings - 1) / weekly_earnings
  weeks_needed = 10

/-- Theorem stating the solution to Alfonso's problem --/
theorem alfonso_solution : 
  alfonso_problem 6 8 460 40 20 := by sorry

end alfonso_solution_l3295_329540


namespace product_of_square_roots_l3295_329571

theorem product_of_square_roots (m : ℝ) :
  Real.sqrt (15 * m) * Real.sqrt (3 * m^2) * Real.sqrt (8 * m^3) = 6 * m^3 * Real.sqrt 10 := by
  sorry

end product_of_square_roots_l3295_329571


namespace tangent_and_inequality_conditions_l3295_329541

noncomputable def f (x : ℝ) := Real.exp (2 * x)
def g (k : ℝ) (x : ℝ) := k * x + 1

theorem tangent_and_inequality_conditions (k : ℝ) :
  (∃ t : ℝ, (f t = g k t ∧ (deriv f) t = k)) ↔ k = 2 ∧
  (k > 0 → (∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, 0 < x → x < m → |f x - g k x| > 2 * x) ↔ k > 4) :=
sorry

end tangent_and_inequality_conditions_l3295_329541


namespace fifth_stack_cups_l3295_329543

def cup_sequence : ℕ → ℕ
  | 0 => 17
  | 1 => 21
  | 2 => 25
  | 3 => 29
  | n + 4 => cup_sequence n + 4

theorem fifth_stack_cups : cup_sequence 4 = 33 := by
  sorry

end fifth_stack_cups_l3295_329543


namespace circle_properties_l3295_329517

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Theorem statement
theorem circle_properties :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    center_x = 3 ∧ center_y = 0 ∧ radius = 3 := by
  sorry

end circle_properties_l3295_329517
