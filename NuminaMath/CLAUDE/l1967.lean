import Mathlib

namespace NUMINAMATH_CALUDE_connors_garage_wheels_l1967_196759

/-- Calculates the total number of wheels in Connor's garage -/
theorem connors_garage_wheels :
  let num_bicycles : ℕ := 20
  let num_cars : ℕ := 10
  let num_motorcycles : ℕ := 5
  let wheels_per_bicycle : ℕ := 2
  let wheels_per_car : ℕ := 4
  let wheels_per_motorcycle : ℕ := 2
  (num_bicycles * wheels_per_bicycle + 
   num_cars * wheels_per_car + 
   num_motorcycles * wheels_per_motorcycle) = 90 := by
  sorry

end NUMINAMATH_CALUDE_connors_garage_wheels_l1967_196759


namespace NUMINAMATH_CALUDE_vectors_form_basis_l1967_196754

def vector_a : Fin 2 → ℝ := λ i => if i = 0 then 2 else -3
def vector_b : Fin 2 → ℝ := λ i => if i = 0 then 6 else 9

def is_basis (v w : Fin 2 → ℝ) : Prop :=
  LinearIndependent ℝ (![v, w]) ∧ Submodule.span ℝ {v, w} = ⊤

theorem vectors_form_basis : is_basis vector_a vector_b := by sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l1967_196754


namespace NUMINAMATH_CALUDE_gymnasts_count_l1967_196717

/-- The number of gymnastics teams --/
def num_teams : ℕ := 4

/-- The total number of handshakes --/
def total_handshakes : ℕ := 595

/-- The number of gymnasts each coach shakes hands with --/
def coach_handshakes : ℕ := 6

/-- The total number of gymnasts across all teams --/
def total_gymnasts : ℕ := 34

/-- Theorem stating that the total number of gymnasts is 34 --/
theorem gymnasts_count : 
  (total_gymnasts * (total_gymnasts - 1)) / 2 + num_teams * coach_handshakes = total_handshakes :=
by sorry

end NUMINAMATH_CALUDE_gymnasts_count_l1967_196717


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1967_196719

theorem quadratic_factorization (C D : ℤ) :
  (∀ x, 15 * x^2 - 56 * x + 48 = (C * x - 8) * (D * x - 6)) →
  C * D + C = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1967_196719


namespace NUMINAMATH_CALUDE_bat_wings_area_l1967_196707

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- Calculates the area of a triangle given three points -/
def triangleArea (a b c : Point) : ℝ :=
  0.5 * abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))

/-- Theorem: The area of the "bat wings" in the given rectangle configuration is 7.5 -/
theorem bat_wings_area (rect : Rectangle)
  (h_width : rect.bottomRight.x - rect.topLeft.x = 4)
  (h_height : rect.bottomRight.y - rect.topLeft.y = 5)
  (j : Point) (k : Point) (l : Point) (m : Point)
  (h_j : j = rect.topLeft)
  (h_k : k.x - j.x = 2 ∧ k.y = rect.bottomRight.y)
  (h_l : l.x = rect.bottomRight.x ∧ l.y - k.y = 2)
  (h_m : m.x = rect.topLeft.x ∧ m.y = rect.bottomRight.y)
  (h_mj : m.y - j.y = 2)
  (h_jk : k.x - j.x = 2)
  (h_kl : l.x - k.x = 2) :
  triangleArea j m k + triangleArea j k l = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_bat_wings_area_l1967_196707


namespace NUMINAMATH_CALUDE_combine_terms_mn_zero_l1967_196753

theorem combine_terms_mn_zero (a b : ℝ) (m n : ℤ) :
  (∃ k : ℝ, ∃ p q : ℤ, -2 * a^m * b^4 + 5 * a^(n+2) * b^(2*m+n) = k * a^p * b^q) →
  m * n = 0 :=
sorry

end NUMINAMATH_CALUDE_combine_terms_mn_zero_l1967_196753


namespace NUMINAMATH_CALUDE_two_thousand_one_in_first_column_l1967_196748

-- Define the column patterns
def first_column (n : ℕ) : Prop := ∃ k : ℕ, n = 16 * k + 1
def second_column (n : ℕ) : Prop := ∃ k : ℕ, n = 16 * k + 3
def third_column (n : ℕ) : Prop := ∃ k : ℕ, n = 16 * k + 5
def fourth_column (n : ℕ) : Prop := ∃ k : ℕ, n = 16 * k + 7

-- Define the theorem
theorem two_thousand_one_in_first_column : 
  first_column 2001 ∧ ¬(second_column 2001 ∨ third_column 2001 ∨ fourth_column 2001) :=
by sorry

end NUMINAMATH_CALUDE_two_thousand_one_in_first_column_l1967_196748


namespace NUMINAMATH_CALUDE_max_ab_value_l1967_196732

noncomputable def g (x : ℝ) : ℝ := 2^x

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : g a * g b = 2) :
  ∀ (x y : ℝ), x > 0 → y > 0 → g x * g y = 2 → x * y ≤ a * b ∧ a * b = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l1967_196732


namespace NUMINAMATH_CALUDE_book_dimensions_and_area_l1967_196740

/-- Represents the dimensions and surface area of a book. -/
structure Book where
  L : ℝ  -- Length
  W : ℝ  -- Width
  T : ℝ  -- Thickness
  A1 : ℝ  -- Area of front cover
  A2 : ℝ  -- Area of spine
  S : ℝ  -- Total surface area

/-- Theorem stating the width and total surface area of a book with given dimensions. -/
theorem book_dimensions_and_area (b : Book) 
  (hL : b.L = 5)
  (hT : b.T = 2)
  (hA1 : b.A1 = 50)
  (hA1_eq : b.A1 = b.L * b.W)
  (hA2_eq : b.A2 = b.T * b.W)
  (hS_eq : b.S = 2 * b.A1 + b.A2 + 2 * (b.L * b.T)) :
  b.W = 10 ∧ b.S = 140 := by
  sorry

#check book_dimensions_and_area

end NUMINAMATH_CALUDE_book_dimensions_and_area_l1967_196740


namespace NUMINAMATH_CALUDE_quadratic_function_m_range_l1967_196705

theorem quadratic_function_m_range
  (a : ℝ) (m : ℝ) (y₁ y₂ : ℝ)
  (h_a_neg : a < 0)
  (h_y₁ : y₁ = a * m^2 - 4 * a * m)
  (h_y₂ : y₂ = a * (2*m)^2 - 4 * a * (2*m))
  (h_above_line : y₁ > -3*a ∧ y₂ > -3*a)
  (h_y₁_gt_y₂ : y₁ > y₂) :
  4/3 < m ∧ m < 3/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_m_range_l1967_196705


namespace NUMINAMATH_CALUDE_binomial_18_4_l1967_196792

theorem binomial_18_4 : Nat.choose 18 4 = 3060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_4_l1967_196792


namespace NUMINAMATH_CALUDE_cubic_three_roots_range_l1967_196728

/-- The cubic polynomial function -/
def f (x : ℝ) := x^3 - 6*x^2 + 9*x

/-- The derivative of f -/
def f' (x : ℝ) := 3*x^2 - 12*x + 9

/-- Theorem: The range of m for which x^3 - 6x^2 + 9x + m = 0 has exactly three distinct real roots is (-4, 0) -/
theorem cubic_three_roots_range :
  ∀ m : ℝ, (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧ 
    f r₁ + m = 0 ∧ f r₂ + m = 0 ∧ f r₃ + m = 0) ↔ 
  -4 < m ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_three_roots_range_l1967_196728


namespace NUMINAMATH_CALUDE_bicycle_distance_l1967_196711

theorem bicycle_distance (back_wheel_perimeter front_wheel_perimeter : ℝ)
  (revolution_difference : ℕ) (distance : ℝ) :
  back_wheel_perimeter = 9 →
  front_wheel_perimeter = 7 →
  revolution_difference = 10 →
  distance / front_wheel_perimeter = distance / back_wheel_perimeter + revolution_difference →
  distance = 315 := by
sorry

end NUMINAMATH_CALUDE_bicycle_distance_l1967_196711


namespace NUMINAMATH_CALUDE_range_of_a_l1967_196776

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |x - 2|

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, f x a ≤ |x - 4|) ↔ a ∈ Set.Icc (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1967_196776


namespace NUMINAMATH_CALUDE_rectangle_area_l1967_196777

theorem rectangle_area (w l : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1967_196777


namespace NUMINAMATH_CALUDE_square_difference_l1967_196761

theorem square_difference (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) :
  a^2 - b^2 = 40 := by sorry

end NUMINAMATH_CALUDE_square_difference_l1967_196761


namespace NUMINAMATH_CALUDE_parabola_properties_l1967_196779

/-- Parabola C: y² = 2px with focus F(2,0) and point A(6,3) -/
def Parabola (p : ℝ) : Set (ℝ × ℝ) :=
  {point : ℝ × ℝ | point.2^2 = 2 * p * point.1}

def F : ℝ × ℝ := (2, 0)
def A : ℝ × ℝ := (6, 3)

/-- The value of p for the given parabola -/
def p_value : ℝ := 4

/-- The minimum value of |MA| + |MF| where M is on the parabola -/
def min_distance : ℝ := 8

theorem parabola_properties :
  ∃ (p : ℝ), p = p_value ∧
  (∀ (M : ℝ × ℝ), M ∈ Parabola p →
    ∀ (d : ℝ), d = Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) + Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) →
    d ≥ min_distance) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l1967_196779


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_m_range_l1967_196787

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The main theorem -/
theorem point_in_fourth_quadrant_m_range (m : ℝ) :
  in_fourth_quadrant ⟨m + 3, m - 5⟩ ↔ -3 < m ∧ m < 5 := by
  sorry


end NUMINAMATH_CALUDE_point_in_fourth_quadrant_m_range_l1967_196787


namespace NUMINAMATH_CALUDE_programming_methods_count_l1967_196700

/-- Represents the number of subprograms -/
def num_subprograms : ℕ := 6

/-- Represents the number of fixed positions (A, B, C, D in order) -/
def fixed_positions : ℕ := 4

/-- Represents the number of remaining subprograms to be placed -/
def remaining_subprograms : ℕ := 2

/-- Represents the number of possible positions for the first remaining subprogram -/
def positions_for_first : ℕ := fixed_positions + 1

/-- Represents the number of possible positions for the second remaining subprogram -/
def positions_for_second : ℕ := fixed_positions + 2

/-- The total number of programming methods -/
def total_methods : ℕ := positions_for_first * positions_for_second

theorem programming_methods_count :
  total_methods = 20 :=
sorry

end NUMINAMATH_CALUDE_programming_methods_count_l1967_196700


namespace NUMINAMATH_CALUDE_bird_ratio_l1967_196746

/-- Proves that the ratio of cardinals to bluebirds is 3:1 given the conditions of the bird problem -/
theorem bird_ratio (cardinals bluebirds swallows : ℕ) 
  (swallow_half : swallows = bluebirds / 2)
  (swallow_count : swallows = 2)
  (total_birds : cardinals + bluebirds + swallows = 18) :
  cardinals = 3 * bluebirds := by
  sorry


end NUMINAMATH_CALUDE_bird_ratio_l1967_196746


namespace NUMINAMATH_CALUDE_largest_angle_in_three_three_four_triangle_l1967_196768

/-- A triangle with interior angles in the ratio 3:3:4 has its largest angle measuring 72° -/
theorem largest_angle_in_three_three_four_triangle : 
  ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  a / 3 = b / 3 ∧ b / 3 = c / 4 →
  max a (max b c) = 72 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_three_three_four_triangle_l1967_196768


namespace NUMINAMATH_CALUDE_middle_managers_sample_size_l1967_196704

/-- Calculates the number of individuals to be selected from a stratum in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (total_sample : ℕ) (stratum_size : ℕ) : ℕ :=
  (total_sample * stratum_size) / total_population

/-- Proves that the number of middle managers to be selected is 6 -/
theorem middle_managers_sample_size :
  stratified_sample_size 160 32 30 = 6 := by
  sorry

#eval stratified_sample_size 160 32 30

end NUMINAMATH_CALUDE_middle_managers_sample_size_l1967_196704


namespace NUMINAMATH_CALUDE_equal_milk_water_ratio_l1967_196773

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- The first mixture with milk:water ratio 5:4 -/
def mixture_p : Mixture := ⟨5, 4⟩

/-- The second mixture with milk:water ratio 2:7 -/
def mixture_q : Mixture := ⟨2, 7⟩

/-- Combines two mixtures in a given ratio -/
def combine_mixtures (m1 m2 : Mixture) (r1 r2 : ℚ) : Mixture :=
  ⟨r1 * m1.milk + r2 * m2.milk, r1 * m1.water + r2 * m2.water⟩

/-- Theorem stating that combining mixture_p and mixture_q in ratio 5:1 results in equal milk and water -/
theorem equal_milk_water_ratio :
  let result := combine_mixtures mixture_p mixture_q 5 1
  result.milk = result.water := by sorry

end NUMINAMATH_CALUDE_equal_milk_water_ratio_l1967_196773


namespace NUMINAMATH_CALUDE_wendy_chocolate_sales_l1967_196706

/-- Calculates the money made from selling chocolate bars -/
def money_made (total_bars : ℕ) (unsold_bars : ℕ) (price_per_bar : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Proves that Wendy made $18 from selling chocolate bars -/
theorem wendy_chocolate_sales : money_made 9 3 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_wendy_chocolate_sales_l1967_196706


namespace NUMINAMATH_CALUDE_julia_age_l1967_196708

-- Define the ages of the individuals
def Grace : ℕ := 20
def Helen : ℕ := Grace + 4
def Ian : ℕ := Helen - 5
def Julia : ℕ := Ian + 2

-- Theorem to prove
theorem julia_age : Julia = 21 := by
  sorry

end NUMINAMATH_CALUDE_julia_age_l1967_196708


namespace NUMINAMATH_CALUDE_reciprocal_of_2023_l1967_196782

theorem reciprocal_of_2023 :
  ∀ (x : ℝ), x = 2023 → x⁻¹ = (1 : ℝ) / 2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_2023_l1967_196782


namespace NUMINAMATH_CALUDE_pen_retailer_profit_percentage_specific_pen_retailer_profit_l1967_196794

/-- Calculates the profit percentage for a retailer selling pens with a discount -/
theorem pen_retailer_profit_percentage 
  (num_pens : ℕ) 
  (price_per_36_pens : ℝ) 
  (discount_percent : ℝ) : ℝ :=
let cost_per_pen := price_per_36_pens / 36
let total_cost := num_pens * cost_per_pen
let selling_price_per_pen := price_per_36_pens / 36 * (1 - discount_percent / 100)
let total_selling_price := num_pens * selling_price_per_pen
let profit := total_selling_price - total_cost
let profit_percentage := (profit / total_cost) * 100
profit_percentage

/-- The profit percentage for a retailer buying 120 pens at the price of 36 pens 
    and selling with a 1% discount is 230% -/
theorem specific_pen_retailer_profit :
  pen_retailer_profit_percentage 120 36 1 = 230 := by
  sorry

end NUMINAMATH_CALUDE_pen_retailer_profit_percentage_specific_pen_retailer_profit_l1967_196794


namespace NUMINAMATH_CALUDE_max_value_problem_1_l1967_196769

theorem max_value_problem_1 (x : ℝ) (h : x < 5/4) :
  ∃ (y : ℝ), y = 4*x - 2 + 1/(4*x - 5) ∧ y ≤ 1 ∧ (∀ (z : ℝ), z = 4*x - 2 + 1/(4*x - 5) → z ≤ y) :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_1_l1967_196769


namespace NUMINAMATH_CALUDE_mixed_groups_count_l1967_196763

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ) 
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  (∃ (mixed_groups : ℕ), 
    mixed_groups = 72 ∧
    mixed_groups * 2 + boy_boy_photos + girl_girl_photos = total_groups * group_size) :=
by sorry


end NUMINAMATH_CALUDE_mixed_groups_count_l1967_196763


namespace NUMINAMATH_CALUDE_smallest_whole_number_larger_than_triangle_perimeter_l1967_196762

theorem smallest_whole_number_larger_than_triangle_perimeter : 
  ∀ s : ℝ, 
  s > 0 → 
  7 + s > 17 → 
  17 + s > 7 → 
  7 + 17 > s → 
  48 > 7 + 17 + s ∧ 
  ∀ n : ℕ, n < 48 → ∃ t : ℝ, t > 0 ∧ 7 + t > 17 ∧ 17 + t > 7 ∧ 7 + 17 > t ∧ n ≤ 7 + 17 + t :=
by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_larger_than_triangle_perimeter_l1967_196762


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_ten_l1967_196715

theorem least_subtraction_for_divisibility_by_ten (n : ℕ) (h : n = 427398) :
  ∃ (k : ℕ), k = 8 ∧ (n - k) % 10 = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_ten_l1967_196715


namespace NUMINAMATH_CALUDE_phoenix_temperature_l1967_196799

theorem phoenix_temperature (t : ℝ) : 
  (∀ s, -s^2 + 14*s + 40 = 77 → s ≤ t) → -t^2 + 14*t + 40 = 77 → t = 11 := by
  sorry

end NUMINAMATH_CALUDE_phoenix_temperature_l1967_196799


namespace NUMINAMATH_CALUDE_tank_capacity_l1967_196786

/-- Represents a cylindrical water tank -/
structure WaterTank where
  capacity : ℝ
  initial_volume : ℝ
  added_volume : ℝ
  final_volume : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem tank_capacity (tank : WaterTank) 
  (h1 : tank.initial_volume = tank.capacity / 6)
  (h2 : tank.added_volume = 4)
  (h3 : tank.final_volume = tank.initial_volume + tank.added_volume)
  (h4 : tank.final_volume = tank.capacity / 5) :
  tank.capacity = 120 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l1967_196786


namespace NUMINAMATH_CALUDE_average_english_score_of_dropped_students_l1967_196724

/-- Represents the problem of calculating the average English quiz score of dropped students -/
theorem average_english_score_of_dropped_students
  (total_students : ℕ)
  (remaining_students : ℕ)
  (initial_average : ℝ)
  (new_average : ℝ)
  (h1 : total_students = 16)
  (h2 : remaining_students = 13)
  (h3 : initial_average = 62.5)
  (h4 : new_average = 62.0) :
  let dropped_students := total_students - remaining_students
  let total_score := total_students * initial_average
  let remaining_score := remaining_students * new_average
  let dropped_score := total_score - remaining_score
  abs ((dropped_score / dropped_students) - 64.67) < 0.01 := by
  sorry

#check average_english_score_of_dropped_students

end NUMINAMATH_CALUDE_average_english_score_of_dropped_students_l1967_196724


namespace NUMINAMATH_CALUDE_triangle_area_proof_l1967_196771

/-- The slope of the first line -/
def m₁ : ℚ := 3/4

/-- The slope of the second line -/
def m₂ : ℚ := -3/2

/-- The x-coordinate of the intersection point of the first two lines -/
def x₀ : ℚ := 4

/-- The y-coordinate of the intersection point of the first two lines -/
def y₀ : ℚ := 3

/-- The equation of the third line: ax + by = c -/
def a : ℚ := 1
def b : ℚ := 2
def c : ℚ := 12

/-- The area of the triangle -/
def triangle_area : ℚ := 15/4

theorem triangle_area_proof :
  let line1 := fun x => m₁ * (x - x₀) + y₀
  let line2 := fun x => m₂ * (x - x₀) + y₀
  let line3 := fun x y => a * x + b * y = c
  ∃ x₁ y₁ x₂ y₂,
    line1 x₁ = y₁ ∧ line3 x₁ y₁ ∧
    line2 x₂ = y₂ ∧ line3 x₂ y₂ ∧
    triangle_area = (1/2) * abs ((x₀ * (y₁ - y₂) + x₁ * (y₂ - y₀) + x₂ * (y₀ - y₁))) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l1967_196771


namespace NUMINAMATH_CALUDE_abs_neg_one_tenth_l1967_196778

theorem abs_neg_one_tenth : |(-1/10 : ℚ)| = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_tenth_l1967_196778


namespace NUMINAMATH_CALUDE_set_equality_l1967_196722

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define the set we want to prove equal to ℂᵤ(M ∪ N)
def target_set : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem set_equality : target_set = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l1967_196722


namespace NUMINAMATH_CALUDE_local_minimum_of_f_l1967_196734

/-- The function f(x) defined as (x^2 + ax - 1)e^(x-1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x - 1) * Real.exp (x - 1)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := (x^2 + x - 2) * Real.exp (x - 1)

theorem local_minimum_of_f (a : ℝ) :
  (f_deriv a (-2) = 0) →  -- x = -2 is a point of extremum
  (∃ (x : ℝ), x > -2 ∧ x < 1 ∧ ∀ y, y > -2 ∧ y < 1 → f a x ≤ f a y) ∧ -- local minimum exists
  (f a 1 = -1) -- the local minimum value is -1
  := by sorry

end NUMINAMATH_CALUDE_local_minimum_of_f_l1967_196734


namespace NUMINAMATH_CALUDE_bike_distance_proof_l1967_196780

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that a bike traveling at 90 km/h for 5 hours covers 450 km -/
theorem bike_distance_proof :
  let speed : ℝ := 90
  let time : ℝ := 5
  distance speed time = 450 := by sorry

end NUMINAMATH_CALUDE_bike_distance_proof_l1967_196780


namespace NUMINAMATH_CALUDE_equilateral_triangle_height_and_area_l1967_196701

/-- Given an equilateral triangle with side length a, prove its height and area. -/
theorem equilateral_triangle_height_and_area (a : ℝ) (h : a > 0) :
  ∃ (height area : ℝ),
    height = (Real.sqrt 3 / 2) * a ∧
    area = (Real.sqrt 3 / 4) * a^2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_height_and_area_l1967_196701


namespace NUMINAMATH_CALUDE_grocer_sales_problem_l1967_196785

theorem grocer_sales_problem (m1 m3 m4 m5 m6 avg : ℕ) (h1 : m1 = 4000) (h3 : m3 = 5689) (h4 : m4 = 7230) (h5 : m5 = 6000) (h6 : m6 = 12557) (havg : avg = 7000) :
  ∃ m2 : ℕ, m2 = 6524 ∧ (m1 + m2 + m3 + m4 + m5 + m6) / 6 = avg :=
sorry

end NUMINAMATH_CALUDE_grocer_sales_problem_l1967_196785


namespace NUMINAMATH_CALUDE_subset_implies_a_geq_one_l1967_196718

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 2*x - 3 < 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x - a < 0}

-- State the theorem
theorem subset_implies_a_geq_one (a : ℝ) : A ⊆ B a → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_geq_one_l1967_196718


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l1967_196709

/-- The lateral surface area of a cylinder with base diameter and height both equal to 2 cm is 4π cm² -/
theorem cylinder_lateral_surface_area :
  ∀ (d h r : ℝ),
  d = 2 →  -- base diameter is 2 cm
  h = 2 →  -- height is 2 cm
  r = d / 2 →  -- radius is half the diameter
  2 * Real.pi * r * h = 4 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l1967_196709


namespace NUMINAMATH_CALUDE_reunion_handshakes_l1967_196766

/-- The number of handshakes when n boys each shake hands once with every other boy -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 9 boys at a reunion, where each boy shakes hands exactly once with each of the others, 
    the total number of handshakes is 36. -/
theorem reunion_handshakes : handshakes 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_reunion_handshakes_l1967_196766


namespace NUMINAMATH_CALUDE_triangle_area_implies_k_difference_l1967_196747

-- Define the lines
def line1 (k₁ b x : ℝ) : ℝ := k₁ * x + 3 * k₁ + b
def line2 (k₂ b x : ℝ) : ℝ := k₂ * x + 3 * k₂ + b

-- Define the theorem
theorem triangle_area_implies_k_difference
  (k₁ k₂ b : ℝ)
  (h1 : k₁ * k₂ < 0)
  (h2 : (1/2) * 3 * |3 * k₁ - 3 * k₂| * 3 = 9) :
  |k₁ - k₂| = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_implies_k_difference_l1967_196747


namespace NUMINAMATH_CALUDE_smallest_palindrome_base2_and_base5_l1967_196760

/-- Checks if a natural number is a palindrome in the given base. -/
def is_palindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def to_base (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- Checks if a natural number has exactly k digits in the given base. -/
def has_k_digits (n : ℕ) (k : ℕ) (base : ℕ) : Prop := sorry

theorem smallest_palindrome_base2_and_base5 :
  ∀ n : ℕ,
  (has_k_digits n 5 2 ∧ is_palindrome n 2 ∧ is_palindrome (to_base n 5).length 5) →
  n ≥ 27 :=
by sorry

end NUMINAMATH_CALUDE_smallest_palindrome_base2_and_base5_l1967_196760


namespace NUMINAMATH_CALUDE_no_solution_exists_l1967_196764

theorem no_solution_exists : ¬∃ x : ℝ, (81 : ℝ)^(3*x) = (27 : ℝ)^(4*x - 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1967_196764


namespace NUMINAMATH_CALUDE_solve_fraction_equation_l1967_196714

theorem solve_fraction_equation :
  ∀ x : ℚ, (2 / 3 : ℚ) - (1 / 4 : ℚ) = 1 / x → x = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_fraction_equation_l1967_196714


namespace NUMINAMATH_CALUDE_remaining_chess_pieces_l1967_196756

def standard_chess_pieces : ℕ := 32
def initial_player_pieces : ℕ := 16
def arianna_lost_pieces : ℕ := 3
def samantha_lost_pieces : ℕ := 9

theorem remaining_chess_pieces :
  standard_chess_pieces - (arianna_lost_pieces + samantha_lost_pieces) = 20 :=
by sorry

end NUMINAMATH_CALUDE_remaining_chess_pieces_l1967_196756


namespace NUMINAMATH_CALUDE_apple_count_equality_l1967_196742

/-- The number of apples Marin has -/
def marins_apples : ℕ := 3

/-- The number of apples David has -/
def davids_apples : ℕ := 3

/-- The difference between Marin's and David's apple counts -/
def apple_difference : ℤ := marins_apples - davids_apples

theorem apple_count_equality : apple_difference = 0 := by
  sorry

end NUMINAMATH_CALUDE_apple_count_equality_l1967_196742


namespace NUMINAMATH_CALUDE_area_ratio_midpoint_quadrilateral_l1967_196788

/-- A quadrilateral in a 2D plane --/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The area of a quadrilateral --/
def area (q : Quadrilateral) : ℝ := sorry

/-- The quadrilateral formed by the midpoints of another quadrilateral's sides --/
def midpointQuadrilateral (q : Quadrilateral) : Quadrilateral := sorry

/-- Theorem: The area of a quadrilateral is twice the area of its midpoint quadrilateral --/
theorem area_ratio_midpoint_quadrilateral (q : Quadrilateral) : 
  area q = 2 * area (midpointQuadrilateral q) := by sorry

end NUMINAMATH_CALUDE_area_ratio_midpoint_quadrilateral_l1967_196788


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l1967_196774

/-- Given two digits A and B in base d > 7 such that AB + AA = 174 in base d,
    prove that A - B = 3 in base d, assuming A > B. -/
theorem digit_difference_in_base_d (d A B : ℕ) : 
  d > 7 →
  A < d →
  B < d →
  A > B →
  (A * d + B) + (A * d + A) = 1 * d * d + 7 * d + 4 →
  A - B = 3 :=
sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l1967_196774


namespace NUMINAMATH_CALUDE_cosine_expression_equals_negative_one_l1967_196726

theorem cosine_expression_equals_negative_one :
  (Real.cos (64 * π / 180) * Real.cos (4 * π / 180) - Real.cos (86 * π / 180) * Real.cos (26 * π / 180)) /
  (Real.cos (71 * π / 180) * Real.cos (41 * π / 180) - Real.cos (49 * π / 180) * Real.cos (19 * π / 180)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_expression_equals_negative_one_l1967_196726


namespace NUMINAMATH_CALUDE_total_orange_purchase_l1967_196738

def initial_purchase : ℕ := 10
def additional_purchase : ℕ := 5
def num_weeks : ℕ := 3
def doubling_weeks : ℕ := 2

theorem total_orange_purchase :
  let first_week := initial_purchase + additional_purchase
  let subsequent_weeks := 2 * first_week * doubling_weeks
  first_week + subsequent_weeks = 75 := by sorry

end NUMINAMATH_CALUDE_total_orange_purchase_l1967_196738


namespace NUMINAMATH_CALUDE_exactly_four_pairs_l1967_196723

/-- A line is tangent to a circle if and only if the distance from the center
    of the circle to the line equals the radius of the circle. -/
def is_tangent_line (m : ℕ) (n : ℕ) : Prop :=
  2^m = 2*n

/-- The condition that n and m are positive integers with n - m < 5 -/
def satisfies_condition (m : ℕ) (n : ℕ) : Prop :=
  0 < m ∧ 0 < n ∧ n < m + 5

/-- The main theorem stating that there are exactly 4 pairs (m, n) satisfying
    both the tangency condition and the inequality condition -/
theorem exactly_four_pairs :
  ∃! (s : Finset (ℕ × ℕ)),
    s.card = 4 ∧
    (∀ (p : ℕ × ℕ), p ∈ s ↔ 
      (is_tangent_line p.1 p.2 ∧ satisfies_condition p.1 p.2)) := by
  sorry

end NUMINAMATH_CALUDE_exactly_four_pairs_l1967_196723


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1967_196729

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (5*x) = 36 → x = 3.6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1967_196729


namespace NUMINAMATH_CALUDE_angle_equivalence_l1967_196781

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
variable (h1 : A.2 = B.2 ∧ B.2 = C.2 ∧ C.2 = D.2)  -- A, B, C, D are on the same line
variable (h2 : A.1 < B.1 ∧ B.1 < C.1 ∧ C.1 < D.1)  -- A, B, C, D are in that order
variable (h3 : dist A B = dist C D)  -- AB = CD
variable (h4 : E.2 ≠ A.2)  -- E is off the line
variable (h5 : dist C E = dist D E)  -- CE = DE

-- Define the angle function
def angle (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem angle_equivalence :
  angle C E D = 2 * angle A E B ↔ dist A C = dist E C :=
sorry

end NUMINAMATH_CALUDE_angle_equivalence_l1967_196781


namespace NUMINAMATH_CALUDE_impossibleToRemoveAllPieces_l1967_196736

/-- Represents the color of a cell or piece -/
inductive Color
| Black
| White

/-- Represents a move on the board -/
structure Move where
  piece1 : Nat × Nat
  piece2 : Nat × Nat
  newPos1 : Nat × Nat
  newPos2 : Nat × Nat

/-- Represents the state of the board -/
structure BoardState where
  pieces : List (Nat × Nat)

/-- Returns the color of a cell given its coordinates -/
def cellColor (pos : Nat × Nat) : Color :=
  if (pos.1 + pos.2) % 2 == 0 then Color.Black else Color.White

/-- Checks if two positions are adjacent -/
def isAdjacent (pos1 pos2 : Nat × Nat) : Bool :=
  (pos1.1 == pos2.1 && (pos1.2 + 1 == pos2.2 || pos1.2 == pos2.2 + 1)) ||
  (pos1.2 == pos2.2 && (pos1.1 + 1 == pos2.1 || pos1.1 == pos2.1 + 1))

/-- Checks if a move is valid -/
def isValidMove (m : Move) : Bool :=
  isAdjacent m.piece1 m.newPos1 && isAdjacent m.piece2 m.newPos2

/-- Applies a move to the board state -/
def applyMove (state : BoardState) (m : Move) : BoardState :=
  sorry

/-- Theorem: It is impossible to remove all pieces from the board -/
theorem impossibleToRemoveAllPieces :
  ∀ (moves : List Move),
    let initialState : BoardState := { pieces := List.range 506 }
    let finalState := moves.foldl applyMove initialState
    finalState.pieces.length > 0 := by
  sorry

end NUMINAMATH_CALUDE_impossibleToRemoveAllPieces_l1967_196736


namespace NUMINAMATH_CALUDE_product_sum_squares_l1967_196789

theorem product_sum_squares (x y : ℝ) :
  x * y = 120 ∧ x^2 + y^2 = 289 → x + y = 22 ∨ x + y = -22 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_squares_l1967_196789


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1967_196750

theorem complex_number_quadrant : ∃ (z : ℂ), z = 2 * Complex.I^3 / (1 - Complex.I) ∧ z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1967_196750


namespace NUMINAMATH_CALUDE_floor_divisibility_l1967_196725

theorem floor_divisibility (n : ℕ) : 
  (2^(n+1) : ℤ) ∣ ⌊((1 : ℝ) + Real.sqrt 3)^(2*n+1)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_divisibility_l1967_196725


namespace NUMINAMATH_CALUDE_x_over_y_value_l1967_196775

theorem x_over_y_value (x y : ℝ) (h : 16 * x = 0.24 * 90 * y) : x / y = 1.35 := by
  sorry

end NUMINAMATH_CALUDE_x_over_y_value_l1967_196775


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l1967_196739

/-- Converts a binary number (represented as a list of bits) to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The given binary numbers -/
def b1 : List Bool := [true, false, true, true]    -- 1101₂
def b2 : List Bool := [true, true, true]           -- 111₂
def b3 : List Bool := [false, true, true, true]    -- 1110₂
def b4 : List Bool := [true, false, false, true]   -- 1001₂
def b5 : List Bool := [false, true, false, true]   -- 1010₂

/-- The result binary number -/
def result : List Bool := [true, false, false, true, true]  -- 11001₂

theorem binary_addition_subtraction :
  binary_to_decimal b1 + binary_to_decimal b2 - binary_to_decimal b3 + 
  binary_to_decimal b4 + binary_to_decimal b5 = binary_to_decimal result := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l1967_196739


namespace NUMINAMATH_CALUDE_english_not_russian_count_l1967_196702

/-- Represents the set of teachers who know English -/
def E : Finset Nat := sorry

/-- Represents the set of teachers who know Russian -/
def R : Finset Nat := sorry

theorem english_not_russian_count :
  (E.card = 75) →
  (R.card = 55) →
  ((E ∩ R).card = 110) →
  ((E \ R).card = 55) := by
  sorry

end NUMINAMATH_CALUDE_english_not_russian_count_l1967_196702


namespace NUMINAMATH_CALUDE_chess_club_boys_l1967_196795

theorem chess_club_boys (total_members : ℕ) (total_attendees : ℕ) :
  total_members = 30 →
  total_attendees = 18 →
  ∃ (boys girls : ℕ),
    boys + girls = total_members ∧
    boys + (2 * girls / 3) = total_attendees ∧
    boys = 6 := by
  sorry

end NUMINAMATH_CALUDE_chess_club_boys_l1967_196795


namespace NUMINAMATH_CALUDE_distance_to_yz_plane_l1967_196730

/-- The distance from a point to the yz-plane -/
def distToYZPlane (x y z : ℝ) : ℝ := |x|

/-- The distance from a point to the x-axis -/
def distToXAxis (x y z : ℝ) : ℝ := |y|

/-- Point P satisfies the given conditions -/
def satisfiesConditions (x y z : ℝ) : Prop :=
  y = -6 ∧ x^2 + z^2 = 36 ∧ distToXAxis x y z = (1/2) * distToYZPlane x y z

theorem distance_to_yz_plane (x y z : ℝ) 
  (h : satisfiesConditions x y z) : distToYZPlane x y z = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_yz_plane_l1967_196730


namespace NUMINAMATH_CALUDE_value_of_a_l1967_196721

theorem value_of_a (a b c : ℤ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 5) 
  (eq3 : c = 3) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l1967_196721


namespace NUMINAMATH_CALUDE_quadratic_polynomial_root_l1967_196720

theorem quadratic_polynomial_root : ∃ (a b c : ℝ), 
  (a = 1) ∧ 
  (∀ x : ℂ, x^2 + b*x + c = 0 ↔ x = 4 + I ∨ x = 4 - I) ∧
  (a*x^2 + b*x + c = x^2 - 8*x + 17) :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_root_l1967_196720


namespace NUMINAMATH_CALUDE_complex_difference_magnitude_l1967_196741

theorem complex_difference_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1)
  (h2 : Complex.abs z₂ = 1)
  (h3 : Complex.abs (z₁ + z₂) = Real.sqrt 3) :
  Complex.abs (z₁ - z₂) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_difference_magnitude_l1967_196741


namespace NUMINAMATH_CALUDE_root_count_theorem_l1967_196758

def count_roots (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry

theorem root_count_theorem (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 - x) = f (2 + x))
  (h2 : ∀ x, f (7 - x) = f (7 + x))
  (h3 : ∀ x ∈ Set.Icc 0 7, f x = 0 ↔ x = 1 ∨ x = 3) :
  count_roots f (-2005) 2005 = 802 :=
sorry

end NUMINAMATH_CALUDE_root_count_theorem_l1967_196758


namespace NUMINAMATH_CALUDE_corn_plants_multiple_of_max_l1967_196749

/-- Represents the number of plants in a garden -/
structure GardenPlants where
  sunflowers : ℕ
  corn : ℕ
  tomatoes : ℕ

/-- Represents the constraints for planting in the garden -/
structure GardenConstraints where
  max_plants_per_row : ℕ
  same_plants_per_row : Bool
  one_type_per_row : Bool

/-- Theorem stating that the number of corn plants must be a multiple of the maximum plants per row -/
theorem corn_plants_multiple_of_max (garden : GardenPlants) (constraints : GardenConstraints) 
  (h1 : garden.sunflowers = 45)
  (h2 : garden.tomatoes = 63)
  (h3 : constraints.max_plants_per_row = 9)
  (h4 : constraints.same_plants_per_row = true)
  (h5 : constraints.one_type_per_row = true) :
  ∃ k : ℕ, garden.corn = k * constraints.max_plants_per_row := by
  sorry

end NUMINAMATH_CALUDE_corn_plants_multiple_of_max_l1967_196749


namespace NUMINAMATH_CALUDE_gcf_and_sum_proof_l1967_196703

def a : ℕ := 198
def b : ℕ := 396

theorem gcf_and_sum_proof : 
  (Nat.gcd a b = a) ∧ 
  (a + 4 * a = 990) := by
sorry

end NUMINAMATH_CALUDE_gcf_and_sum_proof_l1967_196703


namespace NUMINAMATH_CALUDE_greatest_piece_length_l1967_196793

theorem greatest_piece_length (rope1 rope2 rope3 max_length : ℕ) 
  (h1 : rope1 = 48)
  (h2 : rope2 = 72)
  (h3 : rope3 = 120)
  (h4 : max_length = 24) : 
  (Nat.gcd rope1 (Nat.gcd rope2 rope3) ≤ max_length ∧ 
   Nat.gcd rope1 (Nat.gcd rope2 rope3) = max_length) := by
  sorry

#eval Nat.gcd 48 (Nat.gcd 72 120)

end NUMINAMATH_CALUDE_greatest_piece_length_l1967_196793


namespace NUMINAMATH_CALUDE_nearest_integer_to_power_l1967_196798

theorem nearest_integer_to_power : 
  ∃ n : ℤ, |n - (3 + Real.sqrt 5)^6| < 1/2 ∧ n = 2744 :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_power_l1967_196798


namespace NUMINAMATH_CALUDE_square_diff_fourth_power_l1967_196757

theorem square_diff_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_fourth_power_l1967_196757


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1967_196797

theorem unique_solution_condition (m : ℝ) : 
  (∃! x : ℝ, (3 * x + 4) * (x - 8) = -95 + m * x) ↔ 
  (m = -20 - 2 * Real.sqrt 189 ∨ m = -20 + 2 * Real.sqrt 189) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1967_196797


namespace NUMINAMATH_CALUDE_factorial_15_l1967_196737

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_15 : factorial 15 = 1307674368000 := by sorry

end NUMINAMATH_CALUDE_factorial_15_l1967_196737


namespace NUMINAMATH_CALUDE_cashier_payment_problem_l1967_196784

theorem cashier_payment_problem :
  (¬ ∃ x y : ℤ, 72 * x + 105 * y = 1) ∧
  (∃ x y : ℤ, 72 * x + 105 * y = 3) := by
  sorry

end NUMINAMATH_CALUDE_cashier_payment_problem_l1967_196784


namespace NUMINAMATH_CALUDE_alia_markers_count_l1967_196755

theorem alia_markers_count : ∀ (steve austin alia : ℕ),
  steve = 60 →
  austin = steve / 3 →
  alia = 2 * austin →
  alia = 40 := by
sorry

end NUMINAMATH_CALUDE_alia_markers_count_l1967_196755


namespace NUMINAMATH_CALUDE_inequality_proof_l1967_196790

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b)) + (1 / (b - c)) + (4 / (c - a)) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1967_196790


namespace NUMINAMATH_CALUDE_same_color_plate_probability_l1967_196745

theorem same_color_plate_probability : 
  let total_plates : ℕ := 7 + 5
  let red_plates : ℕ := 7
  let blue_plates : ℕ := 5
  let total_combinations : ℕ := Nat.choose total_plates 3
  let red_combinations : ℕ := Nat.choose red_plates 3
  let blue_combinations : ℕ := Nat.choose blue_plates 3
  let same_color_combinations : ℕ := red_combinations + blue_combinations
  (same_color_combinations : ℚ) / total_combinations = 9 / 44 := by
sorry

end NUMINAMATH_CALUDE_same_color_plate_probability_l1967_196745


namespace NUMINAMATH_CALUDE_equation_has_real_root_l1967_196765

/-- The equation x^4 + 2px^3 + x^3 + 2px + 1 = 0 has at least one real root for all real p. -/
theorem equation_has_real_root (p : ℝ) : ∃ x : ℝ, x^4 + 2*p*x^3 + x^3 + 2*p*x + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l1967_196765


namespace NUMINAMATH_CALUDE_nested_expression_simplification_l1967_196713

theorem nested_expression_simplification (x : ℝ) : 1 - (1 - (1 - (1 + (1 - (1 - x))))) = 2 - x := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_simplification_l1967_196713


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_impossible_equal_edge_and_slant_l1967_196751

structure RegularPyramid (n : ℕ) where
  baseEdgeLength : ℝ
  slantHeight : ℝ

/-- Theorem: In a regular hexagonal pyramid, it's impossible for the base edge length
    to be equal to the slant height. -/
theorem hexagonal_pyramid_impossible_equal_edge_and_slant :
  ¬∃ (p : RegularPyramid 6), p.baseEdgeLength = p.slantHeight :=
sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_impossible_equal_edge_and_slant_l1967_196751


namespace NUMINAMATH_CALUDE_trig_identity_proof_l1967_196716

theorem trig_identity_proof : 
  Real.sin (21 * π / 180) * Real.cos (81 * π / 180) - 
  Real.cos (21 * π / 180) * Real.sin (81 * π / 180) = 
  -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l1967_196716


namespace NUMINAMATH_CALUDE_sum_of_squared_distances_l1967_196796

/-- Triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Circumcenter of a triangle -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

/-- A point on the circumcircle of a triangle -/
def point_on_circumcircle (t : Triangle) : ℝ × ℝ := sorry

/-- Squared distance between two points -/
def squared_distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem sum_of_squared_distances (t : Triangle) :
  let O := circumcenter t
  let H := orthocenter t
  let R := circumradius t
  let P := point_on_circumcircle t
  squared_distance P t.A + squared_distance P t.B + squared_distance P t.C + squared_distance P H = 8 * R^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_distances_l1967_196796


namespace NUMINAMATH_CALUDE_transmission_time_is_6_4_minutes_l1967_196770

/-- Represents the number of blocks to be sent -/
def num_blocks : ℕ := 80

/-- Represents the number of chunks in each block -/
def chunks_per_block : ℕ := 768

/-- Represents the transmission rate in chunks per second -/
def transmission_rate : ℕ := 160

/-- Represents the total number of chunks to be sent -/
def total_chunks : ℕ := num_blocks * chunks_per_block

/-- Represents the time in seconds to send all chunks -/
def transmission_time_seconds : ℚ := total_chunks / transmission_rate

/-- Represents the time in minutes to send all chunks -/
def transmission_time_minutes : ℚ := transmission_time_seconds / 60

/-- Theorem stating that the transmission time is 6.4 minutes -/
theorem transmission_time_is_6_4_minutes : transmission_time_minutes = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_transmission_time_is_6_4_minutes_l1967_196770


namespace NUMINAMATH_CALUDE_continuous_function_with_three_preimages_l1967_196772

theorem continuous_function_with_three_preimages :
  ∃ f : ℝ → ℝ, Continuous f ∧
    ∀ y : ℝ, ∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      f x₁ = y ∧ f x₂ = y ∧ f x₃ = y ∧
      ∀ x : ℝ, f x = y → x = x₁ ∨ x = x₂ ∨ x = x₃ :=
by
  sorry

end NUMINAMATH_CALUDE_continuous_function_with_three_preimages_l1967_196772


namespace NUMINAMATH_CALUDE_less_than_reciprocal_check_l1967_196731

def is_less_than_reciprocal (x : ℚ) : Prop :=
  x ≠ 0 ∧ x < 1 / x

theorem less_than_reciprocal_check :
  is_less_than_reciprocal (-3) ∧
  is_less_than_reciprocal (3/4) ∧
  ¬is_less_than_reciprocal (-1/2) ∧
  ¬is_less_than_reciprocal 3 ∧
  ¬is_less_than_reciprocal 0 :=
by sorry

end NUMINAMATH_CALUDE_less_than_reciprocal_check_l1967_196731


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l1967_196733

/-- A regular polygon with side length 8 units and exterior angle 90 degrees has a perimeter of 32 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) : 
  n > 0 ∧ 
  side_length = 8 ∧ 
  exterior_angle = 90 ∧ 
  (360 : ℝ) / n = exterior_angle → 
  n * side_length = 32 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l1967_196733


namespace NUMINAMATH_CALUDE_family_gathering_problem_l1967_196712

theorem family_gathering_problem (P : ℕ) : 
  P / 2 = P - 10 → 
  P / 2 + P / 4 + (P - (P / 2 + P / 4)) = P → 
  P = 20 := by
  sorry

end NUMINAMATH_CALUDE_family_gathering_problem_l1967_196712


namespace NUMINAMATH_CALUDE_parabola_homothety_transform_l1967_196744

/-- A homothety transformation centered at (0,0) with ratio k > 0 -/
structure Homothety where
  k : ℝ
  h_pos : k > 0

/-- The equation of a parabola in the form 2py = x^2 -/
def Parabola (p : ℝ) (x y : ℝ) : Prop := 2 * p * y = x^2

theorem parabola_homothety_transform (p : ℝ) (h_p : p ≠ 0) :
  ∃ (h : Homothety), ∀ (x y : ℝ),
    Parabola p x y ↔ y = x^2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_homothety_transform_l1967_196744


namespace NUMINAMATH_CALUDE_average_temperature_proof_l1967_196727

theorem average_temperature_proof (temp_first_3_days : ℝ) (temp_thur_fri : ℝ) (temp_remaining : ℝ) :
  temp_first_3_days = 40 →
  temp_thur_fri = 80 →
  (3 * temp_first_3_days + 2 * temp_thur_fri + temp_remaining) / 7 = 60 := by
  sorry

#check average_temperature_proof

end NUMINAMATH_CALUDE_average_temperature_proof_l1967_196727


namespace NUMINAMATH_CALUDE_square_difference_eq_85_solutions_l1967_196735

theorem square_difference_eq_85_solutions : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 - p.2^2 = 85) (Finset.product (Finset.range 1000) (Finset.range 1000))).card :=
by
  sorry

end NUMINAMATH_CALUDE_square_difference_eq_85_solutions_l1967_196735


namespace NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l1967_196767

/-- Given an isosceles triangle ABC with area √3/2 and sin(A) = √3 * sin(B),
    prove that the length of one of the legs is √2. -/
theorem isosceles_triangle_leg_length 
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Sides of the triangle
  (h : Real) -- Height of the triangle
  (area : Real) -- Area of the triangle
  (is_isosceles : b = c) -- Triangle is isosceles
  (area_value : area = Real.sqrt 3 / 2) -- Area is √3/2
  (sin_relation : Real.sin A = Real.sqrt 3 * Real.sin B) -- sin(A) = √3 * sin(B)
  : b = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l1967_196767


namespace NUMINAMATH_CALUDE_bella_prob_reach_edge_l1967_196710

/-- Represents a position on the 4x4 grid -/
inductive Position
| Central : Position
| NearEdge : Position
| Edge : Position

/-- Represents the possible directions of movement -/
inductive Direction
| Up | Down | Left | Right

/-- Represents the state of Bella's movement -/
structure BellaState where
  position : Position
  hops : Nat

/-- Transition function for Bella's movement -/
def transition (state : BellaState) (dir : Direction) : BellaState :=
  match state.position with
  | Position.Central => ⟨Position.NearEdge, state.hops + 1⟩
  | Position.NearEdge => 
      if state.hops < 5 then ⟨Position.Edge, state.hops + 1⟩
      else state
  | Position.Edge => state

/-- Probability of reaching an edge square within 5 hops -/
def prob_reach_edge (state : BellaState) : ℚ :=
  sorry

/-- Main theorem: Probability of reaching an edge square within 5 hops is 7/8 -/
theorem bella_prob_reach_edge :
  prob_reach_edge ⟨Position.Central, 0⟩ = 7/8 :=
sorry

end NUMINAMATH_CALUDE_bella_prob_reach_edge_l1967_196710


namespace NUMINAMATH_CALUDE_bottles_per_case_l1967_196791

/-- Represents the number of bottles produced in a day -/
def total_bottles : ℕ := 120000

/-- Represents the number of cases required for one day's production -/
def total_cases : ℕ := 10000

/-- Theorem stating that the number of bottles per case is 12 -/
theorem bottles_per_case :
  total_bottles / total_cases = 12 := by
  sorry

end NUMINAMATH_CALUDE_bottles_per_case_l1967_196791


namespace NUMINAMATH_CALUDE_candy_bar_difference_l1967_196752

/-- Given information about candy bars possessed by Lena, Kevin, and Nicole, 
    prove that Lena has 19.6 more candy bars than Nicole. -/
theorem candy_bar_difference (lena kevin nicole : ℝ) : 
  lena = 37.5 ∧ 
  lena + 9.5 = 5 * kevin ∧ 
  kevin = nicole - 8.5 → 
  lena - nicole = 19.6 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_difference_l1967_196752


namespace NUMINAMATH_CALUDE_triangle_properties_l1967_196783

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition from the problem -/
def satisfiesCondition (t : Triangle) : Prop :=
  (t.b + 2 * t.a) * Real.cos t.C + t.c * Real.cos t.B = 0

/-- The angle bisector of C has length 2 -/
def angleBisectorLength (t : Triangle) : Prop :=
  2 = 2 * t.a * t.b * Real.sin (t.C / 2) / (t.a + t.b)

theorem triangle_properties (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : angleBisectorLength t) : 
  t.C = 2 * Real.pi / 3 ∧ 
  ∀ (a b : ℝ), a > 0 → b > 0 → 2 * a + b ≥ 6 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1967_196783


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1967_196743

/-- The equation of asymptotes for a hyperbola with given parameters -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) (he : Real.sqrt ((a^2 + b^2) / a^2) = 2) :
  ∃ (k : ℝ), k = Real.sqrt 3 ∧ 
  ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) → (y = k * x ∨ y = -k * x) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1967_196743
