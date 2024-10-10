import Mathlib

namespace at_least_one_equation_has_two_distinct_roots_l472_47234

theorem at_least_one_equation_has_two_distinct_roots 
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ x y : ℝ, 
    (x ≠ y ∧ 
      ((a * x^2 + 2*b*x + c = 0 ∧ a * y^2 + 2*b*y + c = 0) ∨
       (b * x^2 + 2*c*x + a = 0 ∧ b * y^2 + 2*c*y + a = 0) ∨
       (c * x^2 + 2*a*x + c = 0 ∧ c * y^2 + 2*a*y + c = 0))) :=
by sorry

end at_least_one_equation_has_two_distinct_roots_l472_47234


namespace cubic_polynomial_root_relation_l472_47255

/-- Given two cubic polynomials h and j, where the roots of j are one less than the roots of h,
    prove that the coefficients of j are (1, 2, 1) -/
theorem cubic_polynomial_root_relation (x : ℝ) :
  let h := fun (x : ℝ) => x^3 - 2*x^2 + 3*x - 1
  let j := fun (x : ℝ) => x^3 + b*x^2 + c*x + d
  (∀ s, h s = 0 → j (s - 1) = 0) →
  (b, c, d) = (1, 2, 1) := by
  sorry

end cubic_polynomial_root_relation_l472_47255


namespace series_convergence_implies_scaled_convergence_l472_47279

theorem series_convergence_implies_scaled_convergence 
  (a : ℕ → ℝ) (h : Summable a) : Summable (fun n => a n / n) := by
  sorry

end series_convergence_implies_scaled_convergence_l472_47279


namespace alice_class_size_l472_47221

/-- The number of students in Alice's white water rafting class -/
def num_students : ℕ := 40

/-- The number of instructors, including Alice -/
def num_instructors : ℕ := 10

/-- The number of life vests Alice has on hand -/
def vests_on_hand : ℕ := 20

/-- The percentage of students bringing their own life vests -/
def percent_students_with_vests : ℚ := 1/5

/-- The additional number of life vests Alice needs to get -/
def additional_vests_needed : ℕ := 22

theorem alice_class_size :
  num_students = 40 ∧
  (num_students + num_instructors) * (1 - percent_students_with_vests) =
    vests_on_hand + additional_vests_needed :=
by sorry

end alice_class_size_l472_47221


namespace heart_ratio_l472_47249

def heart (n m : ℝ) : ℝ := n^4 * m^3

theorem heart_ratio : (heart 3 5) / (heart 5 3) = 3/5 := by
  sorry

end heart_ratio_l472_47249


namespace simplify_expression_l472_47271

theorem simplify_expression (x : ℝ) : 
  Real.sqrt (1 + ((x^6 - 1) / (3 * x^3))^2) = (Real.sqrt (x^12 + 7*x^6 + 1)) / (3 * x^3) :=
by sorry

end simplify_expression_l472_47271


namespace two_triangles_with_perimeter_8_l472_47237

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  perimeter_eq : a + b + c = 8
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

/-- The set of all valid IntTriangles -/
def validTriangles : Set IntTriangle :=
  {t : IntTriangle | t.a > 0 ∧ t.b > 0 ∧ t.c > 0}

/-- Two triangles are considered the same if they have the same multiset of side lengths -/
def sameTriangle (t1 t2 : IntTriangle) : Prop :=
  Multiset.ofList [t1.a, t1.b, t1.c] = Multiset.ofList [t2.a, t2.b, t2.c]

theorem two_triangles_with_perimeter_8 :
    ∃ (t1 t2 : IntTriangle),
      t1 ∈ validTriangles ∧ 
      t2 ∈ validTriangles ∧ 
      ¬(sameTriangle t1 t2) ∧
      ∀ (t : IntTriangle), t ∈ validTriangles → 
        (sameTriangle t t1 ∨ sameTriangle t t2) :=
  sorry

end two_triangles_with_perimeter_8_l472_47237


namespace sphere_volume_after_radius_increase_l472_47285

/-- Given a sphere with initial surface area 400π cm² and radius increased by 2 cm, 
    prove that the new volume is 2304π cm³ -/
theorem sphere_volume_after_radius_increase :
  ∀ (r : ℝ), 
    (4 * π * r^2 = 400 * π) →  -- Initial surface area condition
    ((4 / 3) * π * (r + 2)^3 = 2304 * π) -- New volume after radius increase
:= by sorry

end sphere_volume_after_radius_increase_l472_47285


namespace cubic_function_property_l472_47277

/-- Given a cubic function f(x) = ax³ - bx + 5 where a and b are real numbers,
    if f(-3) = -1, then f(3) = 11. -/
theorem cubic_function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - b * x + 5
  f (-3) = -1 → f 3 = 11 := by
sorry

end cubic_function_property_l472_47277


namespace line_intersects_equidistant_points_in_first_and_second_quadrants_l472_47243

/-- The line equation 4x + 6y = 24 -/
def line_equation (x y : ℝ) : Prop := 4 * x + 6 * y = 24

/-- A point (x, y) is equidistant from coordinate axes if |x| = |y| -/
def equidistant (x y : ℝ) : Prop := abs x = abs y

/-- A point (x, y) is in the first quadrant -/
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- A point (x, y) is in the second quadrant -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The line 4x + 6y = 24 intersects with y = x and y = -x only in the first and second quadrants -/
theorem line_intersects_equidistant_points_in_first_and_second_quadrants :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_equation x₁ y₁ ∧ equidistant x₁ y₁ ∧ first_quadrant x₁ y₁ ∧
    line_equation x₂ y₂ ∧ equidistant x₂ y₂ ∧ second_quadrant x₂ y₂ ∧
    (∀ (x y : ℝ), line_equation x y ∧ equidistant x y →
      first_quadrant x y ∨ second_quadrant x y) :=
by sorry

end line_intersects_equidistant_points_in_first_and_second_quadrants_l472_47243


namespace unique_solution_sum_l472_47276

-- Define the equation
def satisfies_equation (x y : ℕ+) : Prop :=
  (x : ℝ)^2 + 84 * (x : ℝ) + 2008 = (y : ℝ)^2

-- State the theorem
theorem unique_solution_sum :
  ∃! (x y : ℕ+), satisfies_equation x y ∧ x + y = 80 := by
  sorry

end unique_solution_sum_l472_47276


namespace parabola_distance_difference_l472_47260

/-- Parabola type representing y² = 4x -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Function to check if a point is on the parabola -/
def on_parabola (p : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 4 * pt.x

/-- Function to check if a point is on a line -/
def on_line (l : Line) (pt : Point) : Prop :=
  pt.y = l.slope * pt.x + l.intercept

/-- Function to check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- Main theorem -/
theorem parabola_distance_difference 
  (p : Parabola)
  (F N A B : Point)
  (l : Line) :
  p.focus = (1, 0) →
  p.directrix = -1 →
  N.x = -1 ∧ N.y = 0 →
  on_parabola p A →
  on_parabola p B →
  on_line l A →
  on_line l B →
  on_line l F →
  perpendicular (Line.mk (B.y / (B.x - N.x)) 0) l →
  |A.x - F.x| - |B.x - F.x| = 4 :=
sorry

end parabola_distance_difference_l472_47260


namespace stratified_sample_correct_l472_47211

/-- Represents the number of students in each year and the sample size -/
structure SchoolData where
  total_students : ℕ
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ
  sample_size : ℕ

/-- Represents the number of students to be sampled from each year -/
structure SampleAllocation where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Calculates the correct sample allocation for stratified sampling -/
def stratifiedSample (data : SchoolData) : SampleAllocation :=
  { first_year := data.sample_size * data.first_year / data.total_students,
    second_year := data.sample_size * data.second_year / data.total_students,
    third_year := data.sample_size * data.third_year / data.total_students }

/-- Theorem stating that the stratified sampling allocation is correct -/
theorem stratified_sample_correct (data : SchoolData)
  (h1 : data.total_students = 2700)
  (h2 : data.first_year = 900)
  (h3 : data.second_year = 1200)
  (h4 : data.third_year = 600)
  (h5 : data.sample_size = 135) :
  stratifiedSample data = { first_year := 45, second_year := 60, third_year := 30 } :=
by sorry

end stratified_sample_correct_l472_47211


namespace petes_total_distance_l472_47244

/-- Represents the distance Pete traveled in blocks for each leg of his journey -/
structure Journey where
  house_to_garage : ℕ
  garage_to_post_office : ℕ
  post_office_to_friend : ℕ

/-- Calculates the total distance traveled for a round trip -/
def total_distance (j : Journey) : ℕ :=
  2 * (j.house_to_garage + j.garage_to_post_office + j.post_office_to_friend)

/-- Pete's actual journey -/
def petes_journey : Journey :=
  { house_to_garage := 5
  , garage_to_post_office := 20
  , post_office_to_friend := 10 }

/-- Theorem stating that Pete traveled 70 blocks in total -/
theorem petes_total_distance : total_distance petes_journey = 70 := by
  sorry

end petes_total_distance_l472_47244


namespace expression_simplification_and_evaluation_l472_47206

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 7 + 1
  let expr := (x^2 / (x - 3) - 2 * x / (x - 3)) / (x / (x - 3))
  expr = x - 2 ∧ expr = Real.sqrt 7 - 1 := by
  sorry

end expression_simplification_and_evaluation_l472_47206


namespace dice_roll_circle_probability_l472_47212

theorem dice_roll_circle_probability (r : ℕ) (h1 : 3 ≤ r) (h2 : r ≤ 18) :
  2 * Real.pi * r ≤ 2 * Real.pi * r^2 := by sorry

end dice_roll_circle_probability_l472_47212


namespace total_trees_count_l472_47263

/-- Represents the number of Douglas fir trees -/
def D : ℕ := 350

/-- Represents the number of ponderosa pine trees -/
def P : ℕ := 500

/-- The cost of a Douglas fir tree -/
def douglas_cost : ℕ := 300

/-- The cost of a ponderosa pine tree -/
def ponderosa_cost : ℕ := 225

/-- The total cost paid for all trees -/
def total_cost : ℕ := 217500

/-- Theorem stating that given the conditions, the total number of trees is 850 -/
theorem total_trees_count : D + P = 850 ∧ 
  douglas_cost * D + ponderosa_cost * P = total_cost ∧
  (D = 350 ∨ P = 350) := by
  sorry

#check total_trees_count

end total_trees_count_l472_47263


namespace distance_to_origin_of_fourth_point_on_circle_l472_47239

/-- Given four points on a circle, prove that the distance from the fourth point to the origin is √13 -/
theorem distance_to_origin_of_fourth_point_on_circle 
  (A B C D : ℝ × ℝ) 
  (hA : A = (-2, 1)) 
  (hB : B = (-1, 0)) 
  (hC : C = (2, 3)) 
  (hD : D.2 = 3) 
  (h_circle : ∃ (center : ℝ × ℝ) (radius : ℝ), 
    (center.1 - A.1)^2 + (center.2 - A.2)^2 = radius^2 ∧
    (center.1 - B.1)^2 + (center.2 - B.2)^2 = radius^2 ∧
    (center.1 - C.1)^2 + (center.2 - C.2)^2 = radius^2 ∧
    (center.1 - D.1)^2 + (center.2 - D.2)^2 = radius^2) :
  Real.sqrt (D.1^2 + D.2^2) = Real.sqrt 13 := by
  sorry


end distance_to_origin_of_fourth_point_on_circle_l472_47239


namespace g_max_min_sum_l472_47216

def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2*x - 8| + x

theorem g_max_min_sum :
  ∃ (max min : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 10 → g x ≤ max) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 10 ∧ g x = max) ∧
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 10 → min ≤ g x) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 10 ∧ g x = min) ∧
    max + min = 7 :=
by sorry

end g_max_min_sum_l472_47216


namespace intersection_M_N_l472_47253

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - 4*x < 0}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 2} := by sorry

end intersection_M_N_l472_47253


namespace trail_mix_weight_l472_47229

theorem trail_mix_weight : 
  let peanuts : ℚ := 0.16666666666666666
  let chocolate_chips : ℚ := 0.16666666666666666
  let raisins : ℚ := 0.08333333333333333
  let almonds : ℚ := 0.14583333333333331
  let cashews : ℚ := 1/8
  let dried_cranberries : ℚ := 3/32
  peanuts + chocolate_chips + raisins + almonds + cashews + dried_cranberries = 0.78125 := by
  sorry

end trail_mix_weight_l472_47229


namespace sticker_difference_l472_47297

/-- Represents the distribution of stickers in boxes following an arithmetic sequence -/
structure StickerDistribution where
  total : ℕ
  boxes : ℕ
  first : ℕ
  difference : ℕ

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem stating the difference between highest and lowest sticker quantities -/
theorem sticker_difference (dist : StickerDistribution)
  (h1 : dist.total = 250)
  (h2 : dist.boxes = 5)
  (h3 : dist.first = 30)
  (h4 : arithmeticSum dist.first dist.difference dist.boxes = dist.total) :
  dist.first + (dist.boxes - 1) * dist.difference - dist.first = 40 := by
  sorry

#check sticker_difference

end sticker_difference_l472_47297


namespace similar_quadrilateral_longest_side_l472_47204

/-- Given a quadrilateral Q1 with side lengths a, b, c, d, and a similar quadrilateral Q2
    where the minimum side length of Q2 is equal to twice the minimum side length of Q1,
    prove that the longest side of Q2 is twice the longest side of Q1. -/
theorem similar_quadrilateral_longest_side
  (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hmin : a ≤ b ∧ a ≤ c ∧ a ≤ d)
  (hmax : b ≤ d ∧ c ≤ d)
  (h_similar : ∃ (k : ℝ), k > 0 ∧ k * a = 2 * a) :
  ∃ (l : ℝ), l = 2 * d ∧ l = max (k * a) (max (k * b) (max (k * c) (k * d))) :=
sorry

end similar_quadrilateral_longest_side_l472_47204


namespace roof_area_calculation_l472_47203

def roof_area (width : ℝ) (length : ℝ) : ℝ :=
  width * length

theorem roof_area_calculation :
  ∀ w l : ℝ,
  l = 4 * w →
  l - w = 36 →
  roof_area w l = 576 :=
by
  sorry

end roof_area_calculation_l472_47203


namespace no_real_roots_geometric_sequence_l472_47289

/-- If a, b, and c form a geometric sequence, then ax^2 + bx + c = 0 has no real solutions -/
theorem no_real_roots_geometric_sequence (a b c : ℝ) (h1 : a ≠ 0) (h2 : b^2 = a*c) (h3 : a*c > 0) :
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 :=
by sorry

end no_real_roots_geometric_sequence_l472_47289


namespace smallest_three_digit_with_equal_digit_sums_l472_47270

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number satisfies the condition -/
def satisfiesCondition (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≤ n → sumOfDigits n = sumOfDigits (k * n)

/-- Theorem statement -/
theorem smallest_three_digit_with_equal_digit_sums :
  ∃ n : ℕ, n = 999 ∧ 
    (∀ m : ℕ, 100 ≤ m ∧ m < 999 → ¬satisfiesCondition m) ∧
    satisfiesCondition n :=
sorry

end smallest_three_digit_with_equal_digit_sums_l472_47270


namespace initial_number_relation_l472_47288

/-- The game sequence for Professor Célia's number game -/
def game_sequence (n : ℤ) : Vector ℤ 4 :=
  let c := 2 * (n + 1)
  let m := 3 * (c - 1)
  let a := 4 * (m + 1)
  ⟨[n, c, m, a], rfl⟩

/-- Theorem stating the relationship between the initial number and Ademar's number -/
theorem initial_number_relation (n x : ℤ) : 
  (game_sequence n).get 3 = x → n = (x - 16) / 24 :=
sorry

end initial_number_relation_l472_47288


namespace kola_solution_water_added_l472_47298

/-- Represents the composition of a kola solution -/
structure KolaSolution where
  volume : ℝ
  water_percent : ℝ
  kola_percent : ℝ
  sugar_percent : ℝ

def initial_solution : KolaSolution :=
  { volume := 440
  , water_percent := 88
  , kola_percent := 8
  , sugar_percent := 100 - 88 - 8 }

def added_sugar : ℝ := 3.2
def added_kola : ℝ := 6.8
def final_sugar_percent : ℝ := 4.521739130434784

/-- The amount of water added to the solution -/
def water_added : ℝ := 10

theorem kola_solution_water_added :
  let initial_sugar := initial_solution.volume * initial_solution.sugar_percent / 100
  let total_sugar := initial_sugar + added_sugar
  let final_volume := total_sugar / (final_sugar_percent / 100)
  water_added = final_volume - initial_solution.volume - added_sugar - added_kola :=
by sorry

end kola_solution_water_added_l472_47298


namespace area_inscribed_circle_l472_47248

/-- The area of an inscribed circle in a triangle with given side lengths -/
theorem area_inscribed_circle (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let s := (a + b + c) / 2
  let area_triangle := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := area_triangle / s
  π * r^2 = (3136 / 81) * π := by sorry

end area_inscribed_circle_l472_47248


namespace mikes_toy_expenses_l472_47202

/-- The total amount Mike spent on toys -/
def total_spent (marbles_cost football_cost baseball_cost : ℚ) : ℚ :=
  marbles_cost + football_cost + baseball_cost

/-- Theorem stating the total amount Mike spent on toys -/
theorem mikes_toy_expenses :
  total_spent 9.05 4.95 6.52 = 20.52 := by sorry

end mikes_toy_expenses_l472_47202


namespace z_value_and_quadrant_l472_47275

def z : ℂ := (1 + Complex.I) * (3 - 2 * Complex.I)

theorem z_value_and_quadrant :
  z = 5 + Complex.I ∧ Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end z_value_and_quadrant_l472_47275


namespace blue_section_damage_probability_l472_47201

/-- The probability of k successes in n Bernoulli trials -/
def bernoulli_probability (n k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

/-- The number of trials -/
def n : ℕ := 7

/-- The number of successes -/
def k : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

theorem blue_section_damage_probability :
  bernoulli_probability n k p = 128/823543 := by
  sorry

end blue_section_damage_probability_l472_47201


namespace integral_inequality_l472_47241

theorem integral_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a ≤ b) :
  (2 / Real.sqrt 3) * Real.arctan ((2 * (b^2 - a^2)) / ((a^2 + 2) * (b^2 + 2))) ≤
  (∫ (x : ℝ) in a..b, ((x^2 + 1) * (x^2 + x + 1)) / ((x^3 + x^2 + 1) * (x^3 + x + 1))) ∧
  (∫ (x : ℝ) in a..b, ((x^2 + 1) * (x^2 + x + 1)) / ((x^3 + x^2 + 1) * (x^3 + x + 1))) ≤
  (4 / Real.sqrt 3) * Real.arctan (((b - a) * Real.sqrt 3) / (a + b + 2 * (1 + a * b))) :=
by sorry

end integral_inequality_l472_47241


namespace parabola_triangle_area_l472_47231

/-- Given a parabola y = x^2 - 20x + c (c ≠ 0) that intersects the x-axis at points A and B
    and the y-axis at point C, where A and C are symmetrical with respect to the line y = -x,
    the area of triangle ABC is 231. -/
theorem parabola_triangle_area (c : ℝ) (hc : c ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 20*x + c
  let A := (21 : ℝ)
  let B := (-1 : ℝ)
  let C := (0, c)
  (∀ x, f x = 0 → x = A ∨ x = B) →
  (f 0 = c) →
  (A, 0) = (-C.2, 0) →
  (1/2 : ℝ) * (A - B) * (-C.2) = 231 :=
by sorry

end parabola_triangle_area_l472_47231


namespace work_completion_time_l472_47225

theorem work_completion_time 
  (john_time : ℝ) 
  (rose_time : ℝ) 
  (h1 : john_time = 320) 
  (h2 : rose_time = 480) : 
  1 / (1 / john_time + 1 / rose_time) = 192 := by
  sorry

end work_completion_time_l472_47225


namespace prime_sum_squares_l472_47240

theorem prime_sum_squares (p q m : ℕ) : 
  p.Prime → q.Prime → p ≠ q →
  p^2 - 2001*p + m = 0 → q^2 - 2001*q + m = 0 →
  p^2 + q^2 = 3996005 := by
sorry

end prime_sum_squares_l472_47240


namespace circles_properties_l472_47200

-- Define the two circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 4 = 0

-- Define the common chord line
def common_chord (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define the tangent line
def tangent_line (x : ℝ) : Prop := x = -2

theorem circles_properties :
  (∀ x y : ℝ, C1 x y ∧ C2 x y → common_chord x y) ∧
  (∀ x y : ℝ, (C1 x y ∧ tangent_line x) ∨ (C2 x y ∧ tangent_line x) →
    ∃! t : ℝ, (x = -2 ∧ y = t) ∧ (C1 x y ∨ C2 x y)) :=
sorry

end circles_properties_l472_47200


namespace work_absence_problem_l472_47284

theorem work_absence_problem (total_days : ℕ) (daily_wage : ℕ) (daily_fine : ℕ) (total_received : ℕ) :
  total_days = 30 →
  daily_wage = 10 →
  daily_fine = 2 →
  total_received = 216 →
  ∃ (absent_days : ℕ),
    absent_days = 7 ∧
    total_received = daily_wage * (total_days - absent_days) - daily_fine * absent_days :=
by sorry

end work_absence_problem_l472_47284


namespace g_of_4_equals_18_l472_47226

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x - 2

-- Theorem statement
theorem g_of_4_equals_18 : g 4 = 18 := by
  sorry

end g_of_4_equals_18_l472_47226


namespace paths_amc9_count_l472_47286

/-- Represents the number of paths to spell "AMC9" in the grid -/
def pathsAMC9 (m_from_a : Nat) (c_from_m : Nat) (nine_from_c : Nat) : Nat :=
  m_from_a * c_from_m * nine_from_c

/-- Theorem stating that the number of paths to spell "AMC9" is 36 -/
theorem paths_amc9_count :
  pathsAMC9 4 3 3 = 36 := by
  sorry

end paths_amc9_count_l472_47286


namespace dilation_problem_l472_47259

/-- Dilation of a complex number -/
def dilation (z : ℂ) (center : ℂ) (scale : ℝ) : ℂ :=
  center + scale * (z - center)

/-- The problem statement -/
theorem dilation_problem : dilation (-2 + I) (1 - 3*I) 3 = -8 + 9*I := by
  sorry

end dilation_problem_l472_47259


namespace min_product_value_l472_47232

def S (n : ℕ+) : ℚ := n / (n + 1)

def b (n : ℕ+) : ℤ := n - 8

def product (n : ℕ+) : ℚ := (b n : ℚ) * S n

theorem min_product_value :
  ∃ (m : ℕ+), ∀ (n : ℕ+), product m ≤ product n ∧ product m = -4 :=
sorry

end min_product_value_l472_47232


namespace cards_given_to_jeff_l472_47254

/-- The number of cards Nell initially had -/
def initial_cards : ℕ := 455

/-- The number of cards Nell has left -/
def remaining_cards : ℕ := 154

/-- The number of cards Nell gave to Jeff -/
def cards_given : ℕ := initial_cards - remaining_cards

theorem cards_given_to_jeff : cards_given = 301 := by
  sorry

end cards_given_to_jeff_l472_47254


namespace expression_evaluation_l472_47294

theorem expression_evaluation : (900^2 : ℝ) / (306^2 - 294^2) = 112.5 := by
  sorry

end expression_evaluation_l472_47294


namespace simplify_expression_l472_47205

theorem simplify_expression (m : ℝ) : 150*m - 72*m + 3*(5*m) = 93*m := by
  sorry

end simplify_expression_l472_47205


namespace solution_set_a_2_range_of_a_l472_47258

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
sorry

-- Part 2: Range of a when f(x) ≥ 4
theorem range_of_a :
  {a : ℝ | ∃ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} :=
sorry

end solution_set_a_2_range_of_a_l472_47258


namespace sufficient_not_necessary_l472_47273

theorem sufficient_not_necessary (a b x : ℝ) :
  (∀ x, x > a^2 + b^2 → x > 2*a*b) ∧
  (∃ a b x, x > 2*a*b ∧ x ≤ a^2 + b^2) :=
sorry

end sufficient_not_necessary_l472_47273


namespace stuffed_animals_count_l472_47296

/-- The total number of stuffed animals for three girls -/
def total_stuffed_animals (mckenna kenley tenly : ℕ) : ℕ :=
  mckenna + kenley + tenly

/-- Theorem stating the total number of stuffed animals for the three girls -/
theorem stuffed_animals_count :
  ∃ (kenley tenly : ℕ),
    let mckenna := 34
    kenley = 2 * mckenna ∧
    tenly = kenley + 5 ∧
    total_stuffed_animals mckenna kenley tenly = 175 := by
  sorry

end stuffed_animals_count_l472_47296


namespace arithmetic_sequence_and_equation_l472_47267

-- Define arithmetic sequence
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

-- Define the equation from proposition B
def satisfies_equation (a b c : ℝ) : Prop :=
  b ≠ 0 ∧ a / b + c / b = 2

-- Theorem statement
theorem arithmetic_sequence_and_equation :
  (∀ a b c : ℝ, satisfies_equation a b c → is_arithmetic_sequence a b c) ∧
  (∃ a b c : ℝ, is_arithmetic_sequence a b c ∧ ¬satisfies_equation a b c) :=
sorry

end arithmetic_sequence_and_equation_l472_47267


namespace inequality_proof_l472_47209

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) :
  (a / (b^2 + 4)) + (b / (a^2 + 4)) ≥ 1/2 := by
  sorry

end inequality_proof_l472_47209


namespace triangle_inequality_l472_47257

open Real

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b^2 + c^2) / a + (c^2 + a^2) / b + (a^2 + b^2) / c ≥ 2 * (a + b + c) := by
  sorry

end triangle_inequality_l472_47257


namespace quadratic_equation_solutions_l472_47262

theorem quadratic_equation_solutions :
  let eq1 : ℂ → Prop := λ x ↦ x^2 - 6*x + 13 = 0
  let eq2 : ℂ → Prop := λ x ↦ 9*x^2 + 12*x + 29 = 0
  let sol1 : Set ℂ := {3 - 2*I, 3 + 2*I}
  let sol2 : Set ℂ := {-2/3 - 5/3*I, -2/3 + 5/3*I}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ x, eq1 x → x ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ x, eq2 x → x ∈ sol2) := by
  sorry

end quadratic_equation_solutions_l472_47262


namespace cyclic_fraction_inequality_l472_47266

theorem cyclic_fraction_inequality (x y z : ℝ) :
  (x^2 / (x^2 + 2*y*z)) + (y^2 / (y^2 + 2*z*x)) + (z^2 / (z^2 + 2*x*y)) ≥ 1 := by
  sorry

end cyclic_fraction_inequality_l472_47266


namespace mode_and_median_of_data_set_l472_47268

def data_set : List ℕ := [9, 16, 18, 23, 32, 23, 48, 23]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem mode_and_median_of_data_set :
  mode data_set = 23 ∧ median data_set = 23 := by sorry

end mode_and_median_of_data_set_l472_47268


namespace hyperbola_eccentricity_l472_47251

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if it has an asymptote y = √5 x, then its eccentricity is √6. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.sqrt 5) : 
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 6 := by
  sorry

end hyperbola_eccentricity_l472_47251


namespace parabola_equation_proof_l472_47228

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 13 - y^2 / 12 = 1

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (5, 0)

/-- The vertex of the parabola -/
def parabola_vertex : ℝ × ℝ := (0, 0)

/-- The focus of the parabola -/
def parabola_focus : ℝ × ℝ := right_focus

/-- The equation of the parabola -/
def parabola_equation (x y : ℝ) : Prop := y^2 = 20 * x

theorem parabola_equation_proof :
  ∀ x y : ℝ, parabola_equation x y ↔ 
  (parabola_vertex = (0, 0) ∧ parabola_focus = right_focus) :=
sorry

end parabola_equation_proof_l472_47228


namespace cosine_sine_sum_equality_l472_47291

theorem cosine_sine_sum_equality : 
  Real.cos (42 * π / 180) * Real.cos (78 * π / 180) + 
  Real.sin (42 * π / 180) * Real.cos (168 * π / 180) = -1/2 := by
  sorry

end cosine_sine_sum_equality_l472_47291


namespace julie_school_year_work_hours_l472_47278

/-- Julie's summer work scenario -/
structure SummerWork where
  hoursPerWeek : ℕ
  weeks : ℕ
  earnings : ℕ

/-- Julie's school year work scenario -/
structure SchoolYearWork where
  weeks : ℕ
  targetEarnings : ℕ

/-- Calculate required hours per week for school year -/
def requiredHoursPerWeek (summer : SummerWork) (schoolYear : SchoolYearWork) : ℕ :=
  let hourlyWage := summer.earnings / (summer.hoursPerWeek * summer.weeks)
  let totalHours := schoolYear.targetEarnings / hourlyWage
  totalHours / schoolYear.weeks

/-- Theorem: Julie needs to work 15 hours per week during school year -/
theorem julie_school_year_work_hours 
  (summer : SummerWork) 
  (schoolYear : SchoolYearWork) 
  (h1 : summer.hoursPerWeek = 60)
  (h2 : summer.weeks = 10)
  (h3 : summer.earnings = 6000)
  (h4 : schoolYear.weeks = 40)
  (h5 : schoolYear.targetEarnings = 6000) : 
  requiredHoursPerWeek summer schoolYear = 15 := by
  sorry

end julie_school_year_work_hours_l472_47278


namespace ab_max_and_inequality_l472_47213

theorem ab_max_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b + a = 15 - b) :
  (∃ (max : ℝ), max = 9 ∧ a * b ≤ max) ∧ b ≥ 6 - a := by
  sorry

end ab_max_and_inequality_l472_47213


namespace sum_of_roots_quadratic_l472_47217

theorem sum_of_roots_quadratic (a b : ℝ) 
  (ha : a^2 - a - 6 = 0) 
  (hb : b^2 - b - 6 = 0) 
  (hab : a ≠ b) : 
  a + b = 1 := by
  sorry

end sum_of_roots_quadratic_l472_47217


namespace rebuild_points_l472_47281

-- Define a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define symmetry with respect to a point
def symmetric (p1 p2 center : Point) : Prop :=
  center.x = (p1.x + p2.x) / 2 ∧ center.y = (p1.y + p2.y) / 2

theorem rebuild_points (A' B' C' D' : Point) :
  ∃! (A B C D : Point),
    symmetric A A' B ∧
    symmetric B B' C ∧
    symmetric C C' D ∧
    symmetric D D' A :=
  sorry

end rebuild_points_l472_47281


namespace triangle_side_difference_l472_47242

theorem triangle_side_difference (a b : ℕ) (ha : a = 8) (hb : b = 13) : 
  (∃ (x_max x_min : ℕ), 
    (∀ x : ℕ, (x + a > b ∧ x + b > a ∧ a + b > x) → x_min ≤ x ∧ x ≤ x_max) ∧
    (x_max + a > b ∧ x_max + b > a ∧ a + b > x_max) ∧
    (x_min + a > b ∧ x_min + b > a ∧ a + b > x_min) ∧
    (∀ y : ℕ, y > x_max ∨ y < x_min → ¬(y + a > b ∧ y + b > a ∧ a + b > y)) ∧
    x_max - x_min = 14) :=
sorry

end triangle_side_difference_l472_47242


namespace sin_120_degrees_l472_47261

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_120_degrees_l472_47261


namespace missing_number_solution_l472_47235

theorem missing_number_solution : ∃ x : ℤ, (476 + 424) * x - 4 * 476 * 424 = 2704 ∧ x = 904 := by
  sorry

end missing_number_solution_l472_47235


namespace triangle_problem_l472_47233

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.b * Real.cos t.C + t.c * Real.cos t.B = 2 * t.a * Real.cos t.A)
  (h2 : t.b * t.c * Real.cos t.A = Real.sqrt 3) :
  t.A = π / 3 ∧ (1 / 2) * t.b * t.c * Real.sin t.A = 3 / 2 := by
  sorry

end triangle_problem_l472_47233


namespace power_of_power_equals_power_of_product_three_squared_to_fourth_power_l472_47220

theorem power_of_power_equals_power_of_product (a m n : ℕ) :
  (a^m)^n = a^(m*n) :=
sorry

theorem three_squared_to_fourth_power :
  (3^2)^4 = 3^8 ∧ 3^8 = 6561 :=
sorry

end power_of_power_equals_power_of_product_three_squared_to_fourth_power_l472_47220


namespace inscribed_circle_area_ratio_l472_47269

/-- The ratio of the area of an inscribed circle to the area of an equilateral triangle -/
theorem inscribed_circle_area_ratio (s r : ℝ) (h1 : s > 0) (h2 : r > 0) 
  (h3 : r = (Real.sqrt 3 / 6) * s) : 
  (π * r^2) / ((Real.sqrt 3 / 4) * s^2) = π / (3 * Real.sqrt 3) :=
sorry

end inscribed_circle_area_ratio_l472_47269


namespace largest_integer_less_than_sqrt7_plus_sqrt3_power6_l472_47282

theorem largest_integer_less_than_sqrt7_plus_sqrt3_power6 :
  ⌊(Real.sqrt 7 + Real.sqrt 3)^6⌋ = 7039 := by sorry

end largest_integer_less_than_sqrt7_plus_sqrt3_power6_l472_47282


namespace exponent_product_simplification_l472_47227

theorem exponent_product_simplification :
  (5 ^ 0.4) * (5 ^ 0.1) * (5 ^ 0.5) * (5 ^ 0.3) * (5 ^ 0.7) = 25 := by
  sorry

end exponent_product_simplification_l472_47227


namespace complete_square_quadratic_l472_47293

theorem complete_square_quadratic :
  ∀ x : ℝ, x^2 - 4*x - 6 = 0 ↔ (x - 2)^2 = 10 :=
by sorry

end complete_square_quadratic_l472_47293


namespace calculation_proof_l472_47274

theorem calculation_proof : (3.242 * 15) / 100 = 0.4863 := by
  sorry

end calculation_proof_l472_47274


namespace remainder_theorem_l472_47265

theorem remainder_theorem (d r : ℤ) : 
  d > 1 → 
  1059 % d = r →
  1417 % d = r →
  2312 % d = r →
  d - r = 15 := by
sorry

end remainder_theorem_l472_47265


namespace order_of_trig_powers_l472_47218

theorem order_of_trig_powers (α : Real) (h : π/4 < α ∧ α < π/2) :
  (Real.cos α) ^ (Real.sin α) < (Real.cos α) ^ (Real.cos α) ∧
  (Real.cos α) ^ (Real.cos α) < (Real.sin α) ^ (Real.cos α) := by
  sorry

end order_of_trig_powers_l472_47218


namespace tates_education_years_l472_47224

/-- The total years Tate spent in high school and college -/
def total_education_years (normal_hs_duration : ℕ) (hs_reduction : ℕ) (college_multiplier : ℕ) : ℕ :=
  let hs_duration := normal_hs_duration - hs_reduction
  let college_duration := hs_duration * college_multiplier
  hs_duration + college_duration

/-- Theorem stating that Tate's total education years is 12 -/
theorem tates_education_years :
  total_education_years 4 1 3 = 12 := by
  sorry

end tates_education_years_l472_47224


namespace ac_squared_gt_bc_squared_implies_a_gt_b_l472_47247

theorem ac_squared_gt_bc_squared_implies_a_gt_b (a b c : ℝ) (hc : c ≠ 0) :
  a * c^2 > b * c^2 → a > b := by
  sorry

end ac_squared_gt_bc_squared_implies_a_gt_b_l472_47247


namespace d2_equals_18_l472_47299

/-- Definition of E(m) -/
def E (m : ℕ) : ℕ :=
  sorry

/-- The polynomial r(x) -/
def r (x : ℕ) : ℕ :=
  sorry

/-- Theorem stating that d₂ = 18 in the polynomial r(x) that satisfies E(m) = r(m) -/
theorem d2_equals_18 :
  ∃ (d₄ d₃ d₂ d₁ d₀ : ℤ),
    (∀ m : ℕ, m ≥ 7 → Odd m → E m = d₄ * m^4 + d₃ * m^3 + d₂ * m^2 + d₁ * m + d₀) →
    d₂ = 18 :=
  sorry

end d2_equals_18_l472_47299


namespace fraction_sum_equality_l472_47290

theorem fraction_sum_equality : 
  (2 / 20 : ℚ) + (3 / 50 : ℚ) * (5 / 100 : ℚ) + (4 / 1000 : ℚ) + (6 / 10000 : ℚ) = 1076 / 10000 := by
  sorry

end fraction_sum_equality_l472_47290


namespace arithmetic_sequence_common_difference_l472_47252

/-- An arithmetic sequence with common difference d -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | n => a₁ + (n - 1) * d

theorem arithmetic_sequence_common_difference :
  ∀ a₁ : ℝ, ∃ d : ℝ,
    let a := arithmeticSequence a₁ d
    (a 4 = 6) ∧ (a 3 + a 5 = a 10) → d = 1 := by
  sorry

end arithmetic_sequence_common_difference_l472_47252


namespace school_distance_l472_47264

/-- The distance between a child's home and school, given two walking scenarios. -/
theorem school_distance (v₁ v₂ : ℝ) (t₁ t₂ : ℝ) (D : ℝ) : 
  v₁ = 5 →  -- First walking speed in m/min
  v₂ = 7 →  -- Second walking speed in m/min
  t₁ = 6 →  -- Late time in minutes for first scenario
  t₂ = 30 → -- Early time in minutes for second scenario
  v₁ * (D / v₁ + t₁) = D →  -- Equation for first scenario
  v₂ * (D / v₂ - t₂) = D →  -- Equation for second scenario
  D = 630 := by
sorry

end school_distance_l472_47264


namespace max_product_constraint_l472_47295

theorem max_product_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4 * b = 8) :
  a * b ≤ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 4 * b₀ = 8 ∧ a₀ * b₀ = 4 := by
  sorry

end max_product_constraint_l472_47295


namespace multiplication_error_factors_l472_47207

theorem multiplication_error_factors : ∃ (x y z : ℕ), 
  x = y + 10 ∧ 
  x * y = z + 40 ∧ 
  z = 39 * y + 22 ∧ 
  x = 41 ∧ 
  y = 31 := by
sorry

end multiplication_error_factors_l472_47207


namespace train_passing_time_l472_47223

/-- The time it takes for a faster train to completely pass a slower train -/
theorem train_passing_time (v_fast v_slow : ℝ) (length : ℝ) (h_fast : v_fast = 50) (h_slow : v_slow = 32) (h_length : length = 75) :
  (length / ((v_fast - v_slow) * (1000 / 3600))) = 15 :=
sorry

end train_passing_time_l472_47223


namespace probability_red_or_white_l472_47292

/-- Probability of selecting a red or white marble from a bag -/
theorem probability_red_or_white (total : ℕ) (blue : ℕ) (red : ℕ) :
  total = 20 →
  blue = 5 →
  red = 9 →
  (red + (total - blue - red)) / total = 3 / 4 := by
  sorry

end probability_red_or_white_l472_47292


namespace smaller_number_puzzle_l472_47283

theorem smaller_number_puzzle (x y : ℝ) (h_sum : x + y = 18) (h_product : x * y = 80) :
  min x y = 8 := by sorry

end smaller_number_puzzle_l472_47283


namespace inequality_proof_l472_47272

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h : a + b < c + d) : 
  a * c + b * d > a * b := by
sorry

end inequality_proof_l472_47272


namespace factorization_a_squared_minus_one_l472_47256

theorem factorization_a_squared_minus_one (a : ℝ) : a^2 - 1 = (a + 1) * (a - 1) := by
  sorry

end factorization_a_squared_minus_one_l472_47256


namespace time_for_one_smoothie_l472_47219

/-- The time it takes to make a certain number of smoothies -/
def time_to_make_smoothies (n : ℕ) : ℕ := 55

/-- The number of smoothies made in the given time -/
def number_of_smoothies : ℕ := 5

/-- Proves that the time to make one smoothie is 11 minutes -/
theorem time_for_one_smoothie :
  time_to_make_smoothies number_of_smoothies / number_of_smoothies = 11 :=
sorry

end time_for_one_smoothie_l472_47219


namespace binary_representation_properties_l472_47250

/-- A function that counts the number of 1s in the binary representation of a natural number -/
def count_ones (n : ℕ) : ℕ := sorry

/-- A function that counts the number of 0s in the binary representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

/-- Theorem: For any natural number n that is a multiple of 17 and has exactly three 1s in its binary representation:
    1) The binary representation of n has at least six 0s
    2) If the binary representation of n has exactly 7 0s, then n is even -/
theorem binary_representation_properties (n : ℕ) 
  (h1 : n % 17 = 0) 
  (h2 : count_ones n = 3) : 
  (count_zeros n ≥ 6) ∧ 
  (count_zeros n = 7 → Even n) := by sorry

end binary_representation_properties_l472_47250


namespace eat_chips_in_ten_days_l472_47246

/-- The number of days it takes to eat all chips in a bag -/
def days_to_eat_chips (total_chips : ℕ) (first_day_chips : ℕ) (daily_chips : ℕ) : ℕ :=
  1 + (total_chips - first_day_chips) / daily_chips

/-- Theorem: It takes 10 days to eat a bag of 100 chips -/
theorem eat_chips_in_ten_days :
  days_to_eat_chips 100 10 10 = 10 := by
  sorry

end eat_chips_in_ten_days_l472_47246


namespace table_length_proof_l472_47210

/-- Proves that the length of the table is 77 cm given the conditions of the paper placement problem. -/
theorem table_length_proof (table_width : ℕ) (sheet_width sheet_height : ℕ) (x : ℕ) :
  table_width = 80 ∧
  sheet_width = 8 ∧
  sheet_height = 5 ∧
  (x - sheet_height : ℤ) = (table_width - sheet_width : ℤ) →
  x = 77 := by
  sorry

end table_length_proof_l472_47210


namespace points_per_question_l472_47238

theorem points_per_question (first_half : ℕ) (second_half : ℕ) (final_score : ℕ) :
  first_half = 8 →
  second_half = 2 →
  final_score = 80 →
  final_score / (first_half + second_half) = 8 := by
sorry

end points_per_question_l472_47238


namespace books_left_to_read_l472_47230

theorem books_left_to_read (total_books read_books : ℕ) : 
  total_books = 14 → read_books = 8 → total_books - read_books = 6 := by
sorry

end books_left_to_read_l472_47230


namespace exists_positive_solution_l472_47287

/-- Definition of the star operation -/
def star (a b : ℝ) : ℝ := a * b^2 + 3 * b - a

/-- Theorem stating the existence of a positive solution -/
theorem exists_positive_solution :
  ∃ x : ℝ, x > 0 ∧ star 5 x = 100 := by
  sorry

end exists_positive_solution_l472_47287


namespace cistern_emptying_time_l472_47215

theorem cistern_emptying_time (fill_time : ℝ) (combined_fill_time : ℝ) (empty_time : ℝ) : 
  fill_time = 2 → 
  combined_fill_time = 2.571428571428571 →
  (1 / fill_time) - (1 / empty_time) = (1 / combined_fill_time) →
  empty_time = 9 := by
  sorry

end cistern_emptying_time_l472_47215


namespace derivative_of_exp_sin_l472_47208

theorem derivative_of_exp_sin (x : ℝ) :
  deriv (fun x => Real.exp x * Real.sin x) x = Real.exp x * (Real.sin x + Real.cos x) := by
  sorry

end derivative_of_exp_sin_l472_47208


namespace joe_fruit_probability_l472_47245

/-- The number of fruit types Joe can choose from -/
def num_fruit_types : ℕ := 4

/-- The number of meals Joe has in a day -/
def num_meals : ℕ := 4

/-- The probability of choosing a specific fruit for one meal -/
def prob_one_fruit : ℚ := 1 / num_fruit_types

/-- The probability of eating the same fruit for all meals -/
def prob_same_fruit : ℚ := num_fruit_types * (prob_one_fruit ^ num_meals)

/-- The probability of eating at least two different kinds of fruit in one day -/
def prob_different_fruits : ℚ := 1 - prob_same_fruit

theorem joe_fruit_probability : prob_different_fruits = 63 / 64 := by
  sorry

end joe_fruit_probability_l472_47245


namespace solve_system_l472_47222

theorem solve_system (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0)
  (eq1 : x = 2 + 1/z) (eq2 : z = 3 + 1/x) :
  z = (3 + Real.sqrt 15) / 2 := by
  sorry

end solve_system_l472_47222


namespace combined_tax_rate_l472_47236

/-- Combined tax rate calculation -/
theorem combined_tax_rate 
  (john_tax_rate : ℝ) 
  (ingrid_tax_rate : ℝ) 
  (john_income : ℝ) 
  (ingrid_income : ℝ) 
  (h1 : john_tax_rate = 0.30) 
  (h2 : ingrid_tax_rate = 0.40) 
  (h3 : john_income = 56000) 
  (h4 : ingrid_income = 74000) : 
  ∃ (combined_rate : ℝ), 
    combined_rate = (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) :=
by
  sorry

#eval (0.30 * 56000 + 0.40 * 74000) / (56000 + 74000)

end combined_tax_rate_l472_47236


namespace division_remainder_proof_l472_47214

theorem division_remainder_proof (dividend : Nat) (divisor : Nat) (quotient : Nat) (h1 : dividend = 109) (h2 : divisor = 12) (h3 : quotient = 9) :
  dividend % divisor = 1 := by
  sorry

end division_remainder_proof_l472_47214


namespace bianca_cupcakes_theorem_l472_47280

/-- Represents the number of cupcakes Bianca made after selling the first batch -/
def cupcakes_made_after (initial : ℕ) (sold : ℕ) (final : ℕ) : ℕ :=
  final - (initial - sold)

/-- Proves that Bianca made 17 cupcakes after selling the first batch -/
theorem bianca_cupcakes_theorem (initial : ℕ) (sold : ℕ) (final : ℕ)
  (h1 : initial = 14)
  (h2 : sold = 6)
  (h3 : final = 25) :
  cupcakes_made_after initial sold final = 17 := by
  sorry

end bianca_cupcakes_theorem_l472_47280
