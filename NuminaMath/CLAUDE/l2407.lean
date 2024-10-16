import Mathlib

namespace NUMINAMATH_CALUDE_intersection_line_correct_l2407_240779

/-- Two circles in a 2D plane -/
structure TwoCircles where
  c1 : (ℝ × ℝ) → Prop
  c2 : (ℝ × ℝ) → Prop

/-- The equation of a line in 2D -/
structure Line where
  eq : (ℝ × ℝ) → Prop

/-- Given two intersecting circles, returns the line of their intersection -/
def intersectionLine (circles : TwoCircles) : Line :=
  { eq := fun (x, y) => x + 3 * y = 0 }

theorem intersection_line_correct (circles : TwoCircles) :
  circles.c1 = fun (x, y) => x^2 + y^2 = 10 →
  circles.c2 = fun (x, y) => (x - 1)^2 + (y - 3)^2 = 20 →
  ∃ (A B : ℝ × ℝ), circles.c1 A ∧ circles.c1 B ∧ circles.c2 A ∧ circles.c2 B →
  (intersectionLine circles).eq = fun (x, y) => x + 3 * y = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_line_correct_l2407_240779


namespace NUMINAMATH_CALUDE_no_daughters_count_l2407_240788

def berthas_family (num_daughters : ℕ) (total_descendants : ℕ) (daughters_with_children : ℕ) : Prop :=
  num_daughters = 8 ∧
  total_descendants = 40 ∧
  daughters_with_children * 4 = total_descendants - num_daughters

theorem no_daughters_count (num_daughters : ℕ) (total_descendants : ℕ) (daughters_with_children : ℕ) :
  berthas_family num_daughters total_descendants daughters_with_children →
  total_descendants - num_daughters = 32 :=
by sorry

end NUMINAMATH_CALUDE_no_daughters_count_l2407_240788


namespace NUMINAMATH_CALUDE_cookie_baking_time_l2407_240789

/-- Represents the cookie-making process with given times -/
structure CookieProcess where
  total_time : ℕ
  white_icing_time : ℕ
  chocolate_icing_time : ℕ

/-- Calculates the remaining time for batter, baking, and cooling -/
def remaining_time (process : CookieProcess) : ℕ :=
  process.total_time - (process.white_icing_time + process.chocolate_icing_time)

/-- Theorem: The remaining time for batter, baking, and cooling is 60 minutes -/
theorem cookie_baking_time (process : CookieProcess)
    (h1 : process.total_time = 120)
    (h2 : process.white_icing_time = 30)
    (h3 : process.chocolate_icing_time = 30) :
    remaining_time process = 60 := by
  sorry

#eval remaining_time { total_time := 120, white_icing_time := 30, chocolate_icing_time := 30 }

end NUMINAMATH_CALUDE_cookie_baking_time_l2407_240789


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l2407_240719

def total_people : ℕ := 15
def num_men : ℕ := 9
def num_women : ℕ := 6
def committee_size : ℕ := 4

theorem probability_at_least_one_woman :
  let prob_all_men := (num_men / total_people) *
                      ((num_men - 1) / (total_people - 1)) *
                      ((num_men - 2) / (total_people - 2)) *
                      ((num_men - 3) / (total_people - 3))
  (1 : ℚ) - prob_all_men = 59 / 65 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l2407_240719


namespace NUMINAMATH_CALUDE_triangle_properties_l2407_240717

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.b * Real.cos t.A = (2 * t.c + t.a) * Real.cos (Real.pi - t.B) ∧
  t.b = 4 ∧
  (1 / 2) * t.a * t.c * Real.sin t.B = Real.sqrt 3

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  t.B = (2 / 3) * Real.pi ∧ t.a + t.c = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2407_240717


namespace NUMINAMATH_CALUDE_min_tan_product_l2407_240773

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and satisfying bsinC + csinB = 4asinBsinC, the minimum value of tanAtanBtanC is (12 + 7√3) / 3 -/
theorem min_tan_product (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  A + B + C = π →
  b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C →
  (∀ A' B' C' : ℝ,
    0 < A' ∧ A' < π/2 →
    0 < B' ∧ B' < π/2 →
    0 < C' ∧ C' < π/2 →
    A' + B' + C' = π →
    Real.tan A' * Real.tan B' * Real.tan C' ≥ (12 + 7 * Real.sqrt 3) / 3) ∧
  (∃ A' B' C' : ℝ,
    0 < A' ∧ A' < π/2 ∧
    0 < B' ∧ B' < π/2 ∧
    0 < C' ∧ C' < π/2 ∧
    A' + B' + C' = π ∧
    Real.tan A' * Real.tan B' * Real.tan C' = (12 + 7 * Real.sqrt 3) / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_tan_product_l2407_240773


namespace NUMINAMATH_CALUDE_gunther_typing_capacity_l2407_240782

/-- Gunther's typing rate in words per 3 minutes -/
def typing_rate : ℕ := 160

/-- Number of minutes in 3 minutes -/
def minutes_per_unit : ℕ := 3

/-- Number of minutes Gunther works per day -/
def working_minutes : ℕ := 480

/-- Number of words Gunther can type in a working day -/
def words_per_day : ℕ := 25598

theorem gunther_typing_capacity :
  (typing_rate : ℚ) / minutes_per_unit * working_minutes = words_per_day := by
  sorry

end NUMINAMATH_CALUDE_gunther_typing_capacity_l2407_240782


namespace NUMINAMATH_CALUDE_apple_basket_count_l2407_240722

theorem apple_basket_count : 
  ∀ (total : ℕ) (rotten : ℕ) (good : ℕ),
  rotten = (12 * total) / 100 →
  good = 66 →
  good = total - rotten →
  total = 75 := by
sorry

end NUMINAMATH_CALUDE_apple_basket_count_l2407_240722


namespace NUMINAMATH_CALUDE_betty_sugar_purchase_l2407_240759

theorem betty_sugar_purchase (f s : ℝ) : 
  (f ≥ 10 + (3/4) * s) → 
  (f ≤ 3 * s) → 
  (∀ s' : ℝ, (∃ f' : ℝ, f' ≥ 10 + (3/4) * s' ∧ f' ≤ 3 * s') → s' ≥ s) →
  s = 40/9 := by
sorry

end NUMINAMATH_CALUDE_betty_sugar_purchase_l2407_240759


namespace NUMINAMATH_CALUDE_student_guinea_pig_difference_l2407_240706

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 22

/-- The number of guinea pigs in each classroom -/
def guinea_pigs_per_classroom : ℕ := 3

/-- Theorem stating the difference between total students and total guinea pigs -/
theorem student_guinea_pig_difference :
  num_classrooms * students_per_classroom - num_classrooms * guinea_pigs_per_classroom = 95 :=
by sorry

end NUMINAMATH_CALUDE_student_guinea_pig_difference_l2407_240706


namespace NUMINAMATH_CALUDE_no_valid_coloring_l2407_240778

/-- A coloring of a 5x5 board using 4 colors -/
def Coloring := Fin 5 → Fin 5 → Fin 4

/-- Predicate to check if a coloring satisfies the constraint -/
def ValidColoring (c : Coloring) : Prop :=
  ∀ (r1 r2 c1 c2 : Fin 5), r1 ≠ r2 → c1 ≠ c2 →
    (Finset.card {c r1 c1, c r1 c2, c r2 c1, c r2 c2} ≥ 3)

/-- Theorem stating that no valid coloring exists -/
theorem no_valid_coloring : ¬ ∃ (c : Coloring), ValidColoring c := by
  sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l2407_240778


namespace NUMINAMATH_CALUDE_sum_mod_thirteen_l2407_240757

theorem sum_mod_thirteen : (10247 + 10248 + 10249 + 10250) % 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_thirteen_l2407_240757


namespace NUMINAMATH_CALUDE_min_value_2x_3y_l2407_240777

theorem min_value_2x_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y + 3*x*y = 6) :
  ∀ z : ℝ, (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a + 3*b + 3*a*b = 6 ∧ 2*a + 3*b = z) → z ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_3y_l2407_240777


namespace NUMINAMATH_CALUDE_max_boxes_in_wooden_box_l2407_240763

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Converts meters to centimeters -/
def metersToCentimeters (m : ℕ) : ℕ :=
  m * 100

theorem max_boxes_in_wooden_box :
  let largeBox : BoxDimensions :=
    { length := metersToCentimeters 8
      width := metersToCentimeters 7
      height := metersToCentimeters 6 }
  let smallBox : BoxDimensions :=
    { length := 8
      width := 7
      height := 6 }
  (boxVolume largeBox) / (boxVolume smallBox) = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_max_boxes_in_wooden_box_l2407_240763


namespace NUMINAMATH_CALUDE_complex_modulus_l2407_240715

theorem complex_modulus (z : ℂ) : z = (1 + 2*Complex.I)/Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2407_240715


namespace NUMINAMATH_CALUDE_some_number_value_l2407_240774

theorem some_number_value (some_number : ℝ) 
  (h1 : ∃ n : ℝ, (n / 18) * (n / some_number) = 1)
  (h2 : (54 / 18) * (54 / some_number) = 1) : 
  some_number = 162 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l2407_240774


namespace NUMINAMATH_CALUDE_circle_area_difference_l2407_240702

theorem circle_area_difference (π : ℝ) : 
  let r1 : ℝ := 30
  let d2 : ℝ := 30
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 675 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l2407_240702


namespace NUMINAMATH_CALUDE_twice_x_minus_three_greater_than_four_l2407_240748

theorem twice_x_minus_three_greater_than_four (x : ℝ) :
  (2 * x - 3 > 4) ↔ (∃ y, y = 2 * x - 3 ∧ y > 4) :=
sorry

end NUMINAMATH_CALUDE_twice_x_minus_three_greater_than_four_l2407_240748


namespace NUMINAMATH_CALUDE_garden_division_theorem_l2407_240728

/-- Represents a rectangular garden -/
structure Garden where
  width : ℕ
  height : ℕ
  trees : ℕ

/-- Represents a division of the garden -/
structure Division where
  parts : ℕ
  matches_used : ℕ
  trees_per_part : ℕ

/-- Checks if a division is valid for a given garden -/
def is_valid_division (g : Garden) (d : Division) : Prop :=
  d.parts = 4 ∧
  d.matches_used = 12 ∧
  d.trees_per_part * d.parts = g.trees ∧
  d.trees_per_part = 3

theorem garden_division_theorem (g : Garden) 
  (h1 : g.width = 4)
  (h2 : g.height = 3)
  (h3 : g.trees = 12) :
  ∃ d : Division, is_valid_division g d :=
sorry

end NUMINAMATH_CALUDE_garden_division_theorem_l2407_240728


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2407_240745

/-- The eccentricity of an ellipse with major axis length three times its minor axis length -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a = 3 * b) (h5 : a^2 = b^2 + c^2) : c / a = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2407_240745


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l2407_240750

/-- Proves that a price reduction resulting in an 80% increase in sales quantity 
    and a 44% increase in total revenue corresponds to a 20% reduction in price. -/
theorem price_reduction_percentage (P : ℝ) (S : ℝ) (P_new : ℝ) 
  (h1 : P > 0) (h2 : S > 0) (h3 : P_new > 0) :
  (P_new * (S * 1.8) = P * S * 1.44) → (P_new = P * 0.8) :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l2407_240750


namespace NUMINAMATH_CALUDE_part1_part2_l2407_240792

-- Define the quadratic function
def f (b c x : ℝ) : ℝ := x^2 + 2*b*x + c

-- Part 1
theorem part1 (b c : ℝ) : 
  (∀ x, f b c x = 0 ↔ x = -1 ∨ x = 1) → b = 0 ∧ c = -1 := by sorry

-- Part 2
theorem part2 (b : ℝ) :
  (∃ x₁ x₂, f b (b^2 + 2*b + 3) x₁ = 0 ∧ f b (b^2 + 2*b + 3) x₂ = 0 ∧ (x₁ + 1)*(x₂ + 1) = 8) →
  b = -2 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2407_240792


namespace NUMINAMATH_CALUDE_reciprocal_comparison_l2407_240793

theorem reciprocal_comparison :
  let numbers : List ℚ := [1/3, 1/2, 1, 2, 3]
  ∀ x ∈ numbers, x < (1 / x) ↔ (x = 1/3 ∨ x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_comparison_l2407_240793


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2407_240703

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + m*x + 9 = (x + a)^2) → (m = 6 ∨ m = -6) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2407_240703


namespace NUMINAMATH_CALUDE_workday_end_time_l2407_240723

-- Define the start time of the workday
def start_time : Nat := 8

-- Define the lunch break start time
def lunch_start : Nat := 13

-- Define the duration of the workday in hours (excluding lunch)
def workday_duration : Nat := 8

-- Define the duration of the lunch break in hours
def lunch_duration : Nat := 1

-- Theorem to prove the end time of the workday
theorem workday_end_time :
  start_time + workday_duration + lunch_duration = 17 := by
  sorry

#check workday_end_time

end NUMINAMATH_CALUDE_workday_end_time_l2407_240723


namespace NUMINAMATH_CALUDE_divisible_by_24_count_l2407_240732

theorem divisible_by_24_count :
  (∃! (s : Finset ℕ), 
    (∀ a ∈ s, 0 < a ∧ a < 100 ∧ 24 ∣ (a^3 + 23)) ∧ 
    s.card = 5) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_24_count_l2407_240732


namespace NUMINAMATH_CALUDE_final_cost_is_12_l2407_240735

def purchase1 : ℚ := 2.45
def purchase2 : ℚ := 7.60
def purchase3 : ℚ := 3.15
def discount_rate : ℚ := 0.1

def total_before_discount : ℚ := purchase1 + purchase2 + purchase3
def discount_amount : ℚ := total_before_discount * discount_rate
def total_after_discount : ℚ := total_before_discount - discount_amount

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  if x - x.floor < 0.5 then x.floor else x.ceil

theorem final_cost_is_12 :
  round_to_nearest_dollar total_after_discount = 12 := by
  sorry

end NUMINAMATH_CALUDE_final_cost_is_12_l2407_240735


namespace NUMINAMATH_CALUDE_range_of_m_l2407_240766

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ∈ Set.Ioo 1 2 ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2407_240766


namespace NUMINAMATH_CALUDE_gym_distance_proof_l2407_240786

/-- The distance between Wang Lei's home and the gym --/
def gym_distance : ℕ := 1500

/-- Wang Lei's walking speed in meters per minute --/
def wang_lei_speed : ℕ := 40

/-- The older sister's walking speed in meters per minute --/
def older_sister_speed : ℕ := wang_lei_speed + 20

/-- Time taken by the older sister to reach the gym in minutes --/
def time_to_gym : ℕ := 25

/-- Distance from the meeting point to the gym in meters --/
def meeting_point_distance : ℕ := 300

theorem gym_distance_proof :
  gym_distance = older_sister_speed * time_to_gym ∧
  gym_distance = wang_lei_speed * (time_to_gym + meeting_point_distance / wang_lei_speed) :=
by sorry

end NUMINAMATH_CALUDE_gym_distance_proof_l2407_240786


namespace NUMINAMATH_CALUDE_sum_parity_l2407_240758

theorem sum_parity (a b : ℤ) (h : a + b = 1998) : 
  ∃ k : ℤ, 7 * a + 3 * b = 2 * k ∧ 7 * a + 3 * b ≠ 6799 := by
sorry

end NUMINAMATH_CALUDE_sum_parity_l2407_240758


namespace NUMINAMATH_CALUDE_similar_triangle_point_coordinates_l2407_240741

structure Triangle :=
  (O : ℝ × ℝ)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)

def similar_triangle (T1 T2 : Triangle) (ratio : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), 
    (T2.A.1 - center.1 = ratio * (T1.A.1 - center.1)) ∧
    (T2.A.2 - center.2 = ratio * (T1.A.2 - center.2)) ∧
    (T2.B.1 - center.1 = ratio * (T1.B.1 - center.1)) ∧
    (T2.B.2 - center.2 = ratio * (T1.B.2 - center.2))

theorem similar_triangle_point_coordinates 
  (a : ℝ) 
  (OAB : Triangle) 
  (OCD : Triangle) 
  (h1 : OAB.O = (0, 0)) 
  (h2 : OAB.A = (4, 3)) 
  (h3 : OAB.B = (3, a)) 
  (h4 : similar_triangle OAB OCD (1/3)) 
  (h5 : OCD.O = (0, 0)) :
  OCD.A = (4/3, 1) ∨ OCD.A = (-4/3, -1) :=
sorry

end NUMINAMATH_CALUDE_similar_triangle_point_coordinates_l2407_240741


namespace NUMINAMATH_CALUDE_not_divides_power_diff_l2407_240796

theorem not_divides_power_diff (n : ℕ+) : ¬ ∃ k : ℤ, (2^(n : ℕ) + 65) * k = 5^(n : ℕ) - 3^(n : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_diff_l2407_240796


namespace NUMINAMATH_CALUDE_constant_c_value_l2407_240747

theorem constant_c_value (b c : ℚ) :
  (∀ x : ℚ, (x + 3) * (x + b) = x^2 + c*x + 8) →
  c = 17/3 := by
sorry

end NUMINAMATH_CALUDE_constant_c_value_l2407_240747


namespace NUMINAMATH_CALUDE_shorter_side_length_l2407_240725

-- Define the circle and rectangle
def circle_radius : ℝ := 6

-- Define the relationship between circle and rectangle areas
def rectangle_area (circle_area : ℝ) : ℝ := 3 * circle_area

-- Define the theorem
theorem shorter_side_length (circle_area : ℝ) (rectangle_area : ℝ) 
  (h1 : circle_area = π * circle_radius ^ 2)
  (h2 : rectangle_area = 3 * circle_area)
  (h3 : rectangle_area = (2 * circle_radius) * shorter_side) :
  shorter_side = 9 * π := by
  sorry


end NUMINAMATH_CALUDE_shorter_side_length_l2407_240725


namespace NUMINAMATH_CALUDE_segment_ratio_l2407_240729

/-- Given two line segments a and b, where a is 2 meters and b is 40 centimeters,
    prove that the ratio of a to b is 5:1. -/
theorem segment_ratio (a b : ℝ) : a = 2 → b = 40 / 100 → a / b = 5 / 1 := by
  sorry

end NUMINAMATH_CALUDE_segment_ratio_l2407_240729


namespace NUMINAMATH_CALUDE_specific_polyhedron_volume_l2407_240751

/-- A polyhedron formed by folding a flat figure -/
structure Polyhedron where
  /-- The number of equilateral triangles in the flat figure -/
  num_triangles : ℕ
  /-- The number of squares in the flat figure -/
  num_squares : ℕ
  /-- The side length of the squares -/
  square_side : ℝ
  /-- The number of regular hexagons in the flat figure -/
  num_hexagons : ℕ

/-- Calculate the volume of the polyhedron -/
def calculate_volume (p : Polyhedron) : ℝ :=
  sorry

/-- The theorem stating the volume of the specific polyhedron -/
theorem specific_polyhedron_volume :
  let p : Polyhedron := {
    num_triangles := 3,
    num_squares := 3,
    square_side := 2,
    num_hexagons := 1
  }
  calculate_volume p = 11 :=
sorry

end NUMINAMATH_CALUDE_specific_polyhedron_volume_l2407_240751


namespace NUMINAMATH_CALUDE_playground_area_is_4200_l2407_240755

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  length : ℝ
  breadth : ℝ
  playground_area : ℝ

/-- The landscape satisfies the given conditions -/
def is_valid_landscape (l : Landscape) : Prop :=
  l.breadth = 6 * l.length ∧
  l.breadth = 420 ∧
  l.playground_area = (1 / 7) * (l.length * l.breadth)

theorem playground_area_is_4200 (l : Landscape) (h : is_valid_landscape l) :
  l.playground_area = 4200 := by
  sorry

#check playground_area_is_4200

end NUMINAMATH_CALUDE_playground_area_is_4200_l2407_240755


namespace NUMINAMATH_CALUDE_sprint_distance_l2407_240731

def sprint_problem (speed : ℝ) (time : ℝ) : Prop :=
  speed = 6 ∧ time = 4 → speed * time = 24

theorem sprint_distance : sprint_problem 6 4 := by
  sorry

end NUMINAMATH_CALUDE_sprint_distance_l2407_240731


namespace NUMINAMATH_CALUDE_back_sides_average_l2407_240756

def is_prime_or_one (n : ℕ) : Prop := n = 1 ∨ Nat.Prime n

theorem back_sides_average (a b c : ℕ) : 
  is_prime_or_one a ∧ is_prime_or_one b ∧ is_prime_or_one c →
  28 + a = 40 + b ∧ 40 + b = 49 + c →
  (a + b + c) / 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_back_sides_average_l2407_240756


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l2407_240754

/-- Given a parallelogram with opposite vertices at (2, -4) and (14, 10),
    the coordinates of the point where the diagonals intersect are (8, 3). -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -4)
  let v2 : ℝ × ℝ := (14, 10)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (8, 3) := by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l2407_240754


namespace NUMINAMATH_CALUDE_randy_blocks_proof_l2407_240744

theorem randy_blocks_proof (house_blocks tower_blocks : ℕ) 
  (h1 : house_blocks = 89)
  (h2 : tower_blocks = 63)
  (h3 : house_blocks - tower_blocks = 26) :
  house_blocks + tower_blocks = 152 := by
  sorry

end NUMINAMATH_CALUDE_randy_blocks_proof_l2407_240744


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2407_240798

-- Define the inequality function
def f (x : ℝ) : ℝ := |x - 2| * (x - 1)

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x < 2} = Set.Ioi (-Real.pi) ∩ Set.Iio 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2407_240798


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2407_240765

theorem simplify_and_evaluate (a : ℝ) (h : a = 2) : 
  a / (a^2 - 1) - 1 / (a^2 - 1) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2407_240765


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2407_240761

theorem inequality_solution_range (k : ℝ) :
  (∃ x : ℝ, |x + 1| + k < x) ↔ k < -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2407_240761


namespace NUMINAMATH_CALUDE_initial_average_calculation_l2407_240738

theorem initial_average_calculation (n : ℕ) (correct_avg : ℚ) (error : ℚ) : 
  n = 10 → 
  correct_avg = 16 → 
  error = 10 → 
  (n * correct_avg - error) / n = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_calculation_l2407_240738


namespace NUMINAMATH_CALUDE_games_missed_l2407_240776

theorem games_missed (planned_this_month planned_last_month attended : ℕ) 
  (h1 : planned_this_month = 11)
  (h2 : planned_last_month = 17)
  (h3 : attended = 12) :
  planned_this_month + planned_last_month - attended = 16 := by
  sorry

end NUMINAMATH_CALUDE_games_missed_l2407_240776


namespace NUMINAMATH_CALUDE_abc_product_l2407_240797

theorem abc_product (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧
  (∃ k : ℕ, b + 8 = k * a) ∧
  (∃ m n : ℕ, b^2 - 1 = m * a ∧ b^2 - 1 = n * c) ∧
  b + c = a^2 - 1 →
  a * b * c = 2009 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l2407_240797


namespace NUMINAMATH_CALUDE_total_distance_two_trains_l2407_240710

/-- Given two trains A and B traveling for 15 minutes, with speeds of 70 kmph and 90 kmph respectively,
    the total distance covered by both trains is 40 kilometers. -/
theorem total_distance_two_trains (speed_A speed_B : ℝ) (time : ℝ) : 
  speed_A = 70 → speed_B = 90 → time = 0.25 → 
  (speed_A * time + speed_B * time) = 40 := by
sorry

end NUMINAMATH_CALUDE_total_distance_two_trains_l2407_240710


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2407_240791

/-- 
Given a right triangle with sides 6, 8, and 10, and an inscribed square with side length a 
where one vertex of the square coincides with the right angle of the triangle,
and an isosceles right triangle with legs 6 and 6, and an inscribed square with side length b 
where one side of the square lies on the hypotenuse of the triangle,
the ratio of a to b is √2/3.
-/
theorem inscribed_squares_ratio : 
  ∀ (a b : ℝ),
  (∃ (x y : ℝ), x + y = 10 ∧ x^2 + y^2 = 10^2 ∧ x * y = 48 ∧ a * (x - a) = a * (y - a)) →
  (∃ (z : ℝ), z^2 = 72 ∧ b + b = z) →
  a / b = Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2407_240791


namespace NUMINAMATH_CALUDE_jessie_current_weight_l2407_240753

def initial_weight : ℝ := 69
def weight_lost : ℝ := 35

theorem jessie_current_weight : 
  initial_weight - weight_lost = 34 := by sorry

end NUMINAMATH_CALUDE_jessie_current_weight_l2407_240753


namespace NUMINAMATH_CALUDE_inequality_proof_l2407_240713

theorem inequality_proof (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x^2 + y^2 ≤ 1) :
  |x^2 + 2*x*y - y^2| ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2407_240713


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2407_240767

theorem triangle_angle_calculation (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = π) (h5 : Real.sin A ≠ 0) (h6 : Real.sin B ≠ 0) 
  (h7 : 3 / Real.sin A = Real.sqrt 3 / Real.sin B) (h8 : A = π/3) : B = π/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2407_240767


namespace NUMINAMATH_CALUDE_final_card_expectation_l2407_240733

/-- Represents a deck of cards -/
def Deck := List Nat

/-- The process of drawing two cards, discarding one, and reinserting the other -/
def drawDiscardReinsert (d : Deck) : Deck :=
  sorry

/-- The expected value of the label of the remaining card after the process -/
def expectedValue (d : Deck) : Rat :=
  sorry

/-- Theorem stating the expected value of the final card in a 100-card deck -/
theorem final_card_expectation :
  let initialDeck : Deck := List.range 100
  expectedValue initialDeck = 467 / 8 := by
  sorry

end NUMINAMATH_CALUDE_final_card_expectation_l2407_240733


namespace NUMINAMATH_CALUDE_mathematician_paths_l2407_240730

/-- Represents the number of rows in the diagram --/
def num_rows : ℕ := 13

/-- Represents whether the diagram is symmetric --/
def is_symmetric : Prop := true

/-- Represents that each move can be either down-left or down-right --/
def two_move_options : Prop := true

/-- The number of paths spelling "MATHEMATICIAN" in the diagram --/
def num_paths : ℕ := 2^num_rows - 1

theorem mathematician_paths :
  is_symmetric ∧ two_move_options → num_paths = 2^num_rows - 1 := by
  sorry

end NUMINAMATH_CALUDE_mathematician_paths_l2407_240730


namespace NUMINAMATH_CALUDE_orange_ring_weight_l2407_240770

/-- The weight of the orange ring in an experiment, given the weights of other rings and the total weight -/
theorem orange_ring_weight 
  (total_weight : Float) 
  (purple_weight : Float) 
  (white_weight : Float) 
  (h1 : total_weight = 0.8333333333) 
  (h2 : purple_weight = 0.3333333333333333) 
  (h3 : white_weight = 0.4166666666666667) : 
  total_weight - purple_weight - white_weight = 0.0833333333 := by
  sorry

end NUMINAMATH_CALUDE_orange_ring_weight_l2407_240770


namespace NUMINAMATH_CALUDE_money_distribution_l2407_240746

/-- Given the ratios of money between Ram, Gopal, and Krishan, and Ram's amount,
    prove that Krishan has the same amount as Gopal, which is Rs. 1785. -/
theorem money_distribution (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 735 →
  gopal = 1785 ∧ krishan = 1785 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2407_240746


namespace NUMINAMATH_CALUDE_faulty_meter_theorem_l2407_240787

/-- A shopkeeper sells goods using a faulty meter -/
structure Shopkeeper where
  profit_percent : ℝ
  supposed_weight : ℝ
  actual_weight : ℝ

/-- Calculate the weight difference of the faulty meter -/
def faulty_meter_weight (s : Shopkeeper) : ℝ :=
  s.supposed_weight - s.actual_weight

/-- Theorem stating the weight of the faulty meter -/
theorem faulty_meter_theorem (s : Shopkeeper) 
  (h1 : s.profit_percent = 11.11111111111111 / 100)
  (h2 : s.supposed_weight = 1000)
  (h3 : s.actual_weight = (1 - s.profit_percent) * s.supposed_weight) :
  faulty_meter_weight s = 100 := by
  sorry

end NUMINAMATH_CALUDE_faulty_meter_theorem_l2407_240787


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_5_plus_2sqrt6_equality_condition_l2407_240795

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = a*b - 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2*x + y = x*y - 1 → a + 2*b ≤ x + 2*y :=
by sorry

theorem min_value_is_5_plus_2sqrt6 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = a*b - 1) :
  a + 2*b ≥ 5 + 2*Real.sqrt 6 :=
by sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = a*b - 1) :
  a + 2*b = 5 + 2*Real.sqrt 6 ↔ b = 2 + Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_5_plus_2sqrt6_equality_condition_l2407_240795


namespace NUMINAMATH_CALUDE_system_solution_l2407_240712

theorem system_solution (a b c x y z : ℝ) 
  (h1 : x + y + z = 0)
  (h2 : c * x + a * y + b * z = 0)
  (h3 : (x + b)^2 + (y + c)^2 + (z + a)^2 = a^2 + b^2 + c^2)
  (h4 : a ≠ b)
  (h5 : b ≠ c)
  (h6 : a ≠ c) :
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ 
   (x = a - b ∧ y = b - c ∧ z = c - a)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2407_240712


namespace NUMINAMATH_CALUDE_committee_size_l2407_240740

theorem committee_size (n : ℕ) (meetings : ℕ) (members_per_meeting : ℕ)
  (h1 : meetings = 40)
  (h2 : members_per_meeting = 10)
  (h3 : n * (n - 1) / 2 ≥ meetings * (members_per_meeting * (members_per_meeting - 1) / 2)) :
  n > 60 := by
  sorry

end NUMINAMATH_CALUDE_committee_size_l2407_240740


namespace NUMINAMATH_CALUDE_symmetric_points_count_l2407_240781

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if three points are distinct and non-collinear -/
def distinct_non_collinear (M G T : Point2D) : Prop :=
  M ≠ G ∧ M ≠ T ∧ G ≠ T ∧
  (G.x - M.x) * (T.y - M.y) ≠ (T.x - M.x) * (G.y - M.y)

/-- Check if a figure has an axis of symmetry -/
def has_axis_of_symmetry (points : List Point2D) : Prop :=
  sorry  -- Definition omitted for brevity

/-- Count the number of distinct points U that create a figure with symmetry -/
def count_symmetric_points (M G T : Point2D) : ℕ :=
  sorry  -- Definition omitted for brevity

/-- The main theorem -/
theorem symmetric_points_count 
  (M G T : Point2D) 
  (h1 : distinct_non_collinear M G T) 
  (h2 : ¬ has_axis_of_symmetry [M, G, T]) :
  count_symmetric_points M G T = 5 ∨ count_symmetric_points M G T = 6 :=
sorry

end NUMINAMATH_CALUDE_symmetric_points_count_l2407_240781


namespace NUMINAMATH_CALUDE_linear_equation_condition_l2407_240721

theorem linear_equation_condition (a : ℝ) : 
  (|a - 1| = 1 ∧ a - 2 ≠ 0) ↔ a = 0 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l2407_240721


namespace NUMINAMATH_CALUDE_min_value_absolute_sum_l2407_240709

theorem min_value_absolute_sum (x : ℝ) : 
  |x - 4| + |x + 8| + |x - 5| ≥ -25 ∧ ∃ y : ℝ, |y - 4| + |y + 8| + |y - 5| = -25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_absolute_sum_l2407_240709


namespace NUMINAMATH_CALUDE_monomial_sum_implies_a_power_l2407_240700

/-- Given two monomials in x and y whose sum is a monomial, prove a^2004 - 1 = 0 --/
theorem monomial_sum_implies_a_power (m n : ℤ) (a : ℕ) :
  (∃ (x y : ℝ), (3 * m * x^a * y) + (-2 * n * x^(4*a - 3) * y) = x^k * y) →
  a^2004 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_monomial_sum_implies_a_power_l2407_240700


namespace NUMINAMATH_CALUDE_matrix_sum_theorem_l2407_240749

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -1; 3, 7]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 8; 5, -2]

theorem matrix_sum_theorem : A + B = !![(-2), 7; 8, 5] := by sorry

end NUMINAMATH_CALUDE_matrix_sum_theorem_l2407_240749


namespace NUMINAMATH_CALUDE_f_properties_l2407_240743

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - a / (3^x + 1)

theorem f_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 2 ∧ ∀ x y, x < y → f 2 x < f 2 y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2407_240743


namespace NUMINAMATH_CALUDE_unique_valid_number_l2407_240714

def is_valid_number (n : ℕ) : Prop :=
  (∀ k : ℕ, k ∈ Finset.range 10 → (n / 10^(9-k)) % k = 0) ∧
  (∀ d : ℕ, d ∈ Finset.range 10 → (∃! i : ℕ, i ∈ Finset.range 9 ∧ (n / 10^i) % 10 = d))

theorem unique_valid_number :
  ∃! n : ℕ, n = 381654729 ∧ is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l2407_240714


namespace NUMINAMATH_CALUDE_next_square_property_l2407_240769

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_square_property (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  is_perfect_square ((n / 100) * (n % 100))

theorem next_square_property : 
  ∀ n : ℕ, n > 1818 → has_square_property n → n ≥ 1832 :=
sorry

end NUMINAMATH_CALUDE_next_square_property_l2407_240769


namespace NUMINAMATH_CALUDE_mrs_lee_june_earnings_percent_l2407_240780

/-- Represents the Lee family's income situation -/
structure LeeIncome where
  may_total : ℝ
  may_mrs_lee : ℝ
  june_mrs_lee : ℝ

/-- Conditions for the Lee family's income -/
def lee_income_conditions (income : LeeIncome) : Prop :=
  income.may_mrs_lee = 0.5 * income.may_total ∧
  income.june_mrs_lee = 1.2 * income.may_mrs_lee

/-- Theorem: Mrs. Lee's earnings in June were 60% of the family's total income -/
theorem mrs_lee_june_earnings_percent (income : LeeIncome) 
  (h : lee_income_conditions income) : 
  income.june_mrs_lee / income.may_total = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_mrs_lee_june_earnings_percent_l2407_240780


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l2407_240704

theorem sarahs_bowling_score (jessica greg sarah : ℕ) 
  (h1 : sarah = greg + 50)
  (h2 : greg = 2 * jessica)
  (h3 : (sarah + greg + jessica) / 3 = 110) :
  sarah = 162 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l2407_240704


namespace NUMINAMATH_CALUDE_no_base_square_l2407_240724

theorem no_base_square (b : ℕ) : b > 1 → ¬∃ (n : ℕ), 2 * b^2 + 3 * b + 2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_base_square_l2407_240724


namespace NUMINAMATH_CALUDE_clean_city_workers_l2407_240799

/-- The number of people in Lizzie's group -/
def lizzies_group : ℕ := 54

/-- The difference in members between Lizzie's group and the other group -/
def difference : ℕ := 17

/-- The total number of people working together to clean the city -/
def total_people : ℕ := lizzies_group + (lizzies_group - difference)

/-- Theorem stating that the total number of people working together is 91 -/
theorem clean_city_workers : total_people = 91 := by sorry

end NUMINAMATH_CALUDE_clean_city_workers_l2407_240799


namespace NUMINAMATH_CALUDE_distance_difference_l2407_240768

/-- Represents the distance traveled by a biker given their speed and time. -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents Camila's constant speed in miles per hour. -/
def camila_speed : ℝ := 15

/-- Represents Daniel's initial speed in miles per hour. -/
def daniel_initial_speed : ℝ := 15

/-- Represents Daniel's reduced speed in miles per hour. -/
def daniel_reduced_speed : ℝ := 10

/-- Represents the total time of the bike ride in hours. -/
def total_time : ℝ := 6

/-- Represents the time at which Daniel's speed changes in hours. -/
def speed_change_time : ℝ := 3

/-- Calculates the distance Camila travels in 6 hours. -/
def camila_distance : ℝ := distance camila_speed total_time

/-- Calculates the distance Daniel travels in 6 hours. -/
def daniel_distance : ℝ := 
  distance daniel_initial_speed speed_change_time + 
  distance daniel_reduced_speed (total_time - speed_change_time)

theorem distance_difference : camila_distance - daniel_distance = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l2407_240768


namespace NUMINAMATH_CALUDE_value_multiplied_with_b_l2407_240775

theorem value_multiplied_with_b (a b x : ℚ) : 
  a / b = 6 / 5 → 
  (5 * a + x * b) / (5 * a - x * b) = 5 →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_value_multiplied_with_b_l2407_240775


namespace NUMINAMATH_CALUDE_sine_inequality_l2407_240771

theorem sine_inequality (x : Real) (h : 0 < x ∧ x < Real.pi / 4) :
  Real.sin (Real.sin x) < Real.sin x ∧ Real.sin x < Real.sin (Real.tan x) :=
by sorry

end NUMINAMATH_CALUDE_sine_inequality_l2407_240771


namespace NUMINAMATH_CALUDE_product_of_numbers_l2407_240794

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2407_240794


namespace NUMINAMATH_CALUDE_quadratic_root_property_l2407_240752

theorem quadratic_root_property (m : ℝ) : m^2 - m - 3 = 0 → 2023 - m^2 + m = 2020 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l2407_240752


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2407_240726

theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → Prop := fun x y => x^2 / 4 - y^2 / 9 = 1
  ∀ x y : ℝ, (∃ t : ℝ, t ≠ 0 ∧ h (t * x) (t * y)) ↔ y = (3/2) * x ∨ y = -(3/2) * x :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2407_240726


namespace NUMINAMATH_CALUDE_point_on_y_axis_l2407_240772

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of a point being on the y-axis -/
def on_y_axis (p : CartesianPoint) : Prop := p.x = 0

/-- Theorem: A point with x-coordinate 0 lies on the y-axis -/
theorem point_on_y_axis (p : CartesianPoint) (h : p.x = 0) : on_y_axis p := by
  sorry

#check point_on_y_axis

end NUMINAMATH_CALUDE_point_on_y_axis_l2407_240772


namespace NUMINAMATH_CALUDE_lake_view_population_l2407_240784

theorem lake_view_population (seattle boise lakeview : ℕ) : 
  boise = (3 * seattle) / 5 →
  lakeview = seattle + 4000 →
  boise + seattle + lakeview = 56000 →
  lakeview = 24000 := by
sorry

end NUMINAMATH_CALUDE_lake_view_population_l2407_240784


namespace NUMINAMATH_CALUDE_pizza_diameter_increase_l2407_240742

theorem pizza_diameter_increase (d : ℝ) (D : ℝ) (h : d > 0) (h' : D > 0) :
  (π * (D / 2)^2 = 1.96 * π * (d / 2)^2) →
  (D = 1.4 * d) := by
sorry

end NUMINAMATH_CALUDE_pizza_diameter_increase_l2407_240742


namespace NUMINAMATH_CALUDE_pencils_per_box_l2407_240762

/-- The number of pencils Louise has for each color and the number of boxes --/
structure PencilData where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ
  boxes : ℕ

/-- The conditions of Louise's pencil organization --/
def validPencilData (d : PencilData) : Prop :=
  d.red = 20 ∧
  d.blue = 2 * d.red ∧
  d.yellow = 40 ∧
  d.green = d.red + d.blue ∧
  d.boxes = 8

/-- The theorem stating that each box holds 20 pencils --/
theorem pencils_per_box (d : PencilData) (h : validPencilData d) :
  (d.red + d.blue + d.yellow + d.green) / d.boxes = 20 := by
  sorry

#check pencils_per_box

end NUMINAMATH_CALUDE_pencils_per_box_l2407_240762


namespace NUMINAMATH_CALUDE_stock_price_change_l2407_240705

theorem stock_price_change (total_stocks : ℕ) 
  (h1 : total_stocks = 1980)
  (h2 : ∀ s : Fin total_stocks, ∃ (price_yesterday price_today : ℝ), price_yesterday ≠ price_today)
  (h3 : ∃ (higher lower : ℕ), 
    higher + lower = total_stocks ∧ 
    higher = lower + (lower / 5)) :
  ∃ (higher : ℕ), higher = 1080 ∧ 
    ∃ (lower : ℕ), higher + lower = total_stocks ∧ 
    higher = lower + (lower / 5) := by
sorry

end NUMINAMATH_CALUDE_stock_price_change_l2407_240705


namespace NUMINAMATH_CALUDE_routes_between_plains_cities_l2407_240716

theorem routes_between_plains_cities 
  (total_cities : Nat) 
  (mountainous_cities : Nat) 
  (plains_cities : Nat) 
  (total_routes : Nat) 
  (mountainous_routes : Nat) : 
  total_cities = 100 → 
  mountainous_cities = 30 → 
  plains_cities = 70 → 
  total_routes = 150 → 
  mountainous_routes = 21 → 
  ∃ (plains_routes : Nat), plains_routes = 81 ∧ 
    plains_routes + mountainous_routes + (total_routes - plains_routes - mountainous_routes) = total_routes := by
  sorry

end NUMINAMATH_CALUDE_routes_between_plains_cities_l2407_240716


namespace NUMINAMATH_CALUDE_always_two_distinct_roots_find_p_values_l2407_240764

-- Define the quadratic equation
def quadratic_equation (x p : ℝ) : ℝ := (x - 3) * (x - 2) - p^2

-- Part 1: Prove that the equation always has two distinct real roots
theorem always_two_distinct_roots (p : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ p = 0 ∧ quadratic_equation x₂ p = 0 := by
  sorry

-- Part 2: Find the values of p given the condition x₁ = 4x₂
theorem find_p_values :
  ∃ p : ℝ, ∃ x₁ x₂ : ℝ, 
    quadratic_equation x₁ p = 0 ∧ 
    quadratic_equation x₂ p = 0 ∧ 
    x₁ = 4 * x₂ ∧ 
    (p = Real.sqrt 2 ∨ p = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_always_two_distinct_roots_find_p_values_l2407_240764


namespace NUMINAMATH_CALUDE_min_width_rectangle_l2407_240739

/-- Given a rectangular area of at least 200 sq. ft. with length 10 ft shorter than twice the width, 
    the minimum width is 10 feet. -/
theorem min_width_rectangle (w : ℝ) (h1 : w > 0) : 
  (w * (2 * w - 10) ≥ 200) → w ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_min_width_rectangle_l2407_240739


namespace NUMINAMATH_CALUDE_line_equation_proof_l2407_240790

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - y + 3 = 0
def line2 (x y : ℝ) : Prop := 4*x + 3*y + 1 = 0
def line3 (x y : ℝ) : Prop := 2*x - 3*y + 4 = 0
def line_result (x y : ℝ) : Prop := 3*x + 2*y + 1 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem line_equation_proof :
  ∃ x y : ℝ, 
    intersection_point x y ∧ 
    line_result x y ∧
    perpendicular (3/2) (-2/3) :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2407_240790


namespace NUMINAMATH_CALUDE_binomial_2586_1_l2407_240708

theorem binomial_2586_1 : Nat.choose 2586 1 = 2586 := by sorry

end NUMINAMATH_CALUDE_binomial_2586_1_l2407_240708


namespace NUMINAMATH_CALUDE_base7_5463_equals_1956_l2407_240707

def base7ToBase10 (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

theorem base7_5463_equals_1956 : base7ToBase10 5 4 6 3 = 1956 := by
  sorry

end NUMINAMATH_CALUDE_base7_5463_equals_1956_l2407_240707


namespace NUMINAMATH_CALUDE_linear_function_point_values_l2407_240720

theorem linear_function_point_values (a m n b : ℝ) :
  (∃ (m n : ℝ), n = 2 * m + b ∧ a = 2 * (1/2) + b) →
  (∀ (m n : ℝ), n = 2 * m + b → m * n ≥ -8) →
  (∃ (m n : ℝ), n = 2 * m + b ∧ m * n = -8) →
  (a = -7 ∨ a = 9) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_point_values_l2407_240720


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2407_240727

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2407_240727


namespace NUMINAMATH_CALUDE_units_digit_of_seven_power_l2407_240718

theorem units_digit_of_seven_power : ∃ n : ℕ, 7^(6^5) ≡ 1 [ZMOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_power_l2407_240718


namespace NUMINAMATH_CALUDE_circle_construction_theorem_l2407_240783

-- Define the basic structures
structure Point :=
  (x y : ℝ)

structure Line :=
  (a b c : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define the given elements
variable (P : Point)
variable (L1 L2 : Line)

-- Define the property of a circle touching a line
def CircleTouchesLine (C : Circle) (L : Line) : Prop :=
  sorry

-- Define the property of a point lying on a circle
def PointOnCircle (P : Point) (C : Circle) : Prop :=
  sorry

-- Theorem statement
theorem circle_construction_theorem :
  ∃ (C1 C2 : Circle),
    (PointOnCircle P C1 ∧ CircleTouchesLine C1 L1 ∧ CircleTouchesLine C1 L2) ∧
    (PointOnCircle P C2 ∧ CircleTouchesLine C2 L1 ∧ CircleTouchesLine C2 L2) :=
by sorry

end NUMINAMATH_CALUDE_circle_construction_theorem_l2407_240783


namespace NUMINAMATH_CALUDE_complement_intersection_M_N_l2407_240785

def U : Set Nat := {1, 2, 3, 4}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {2, 3, 4}

theorem complement_intersection_M_N :
  (U \ (M ∩ N)) = {1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_M_N_l2407_240785


namespace NUMINAMATH_CALUDE_school_walk_time_difference_l2407_240760

/-- Proves that a child walking to school is 6 minutes late when walking at 5 m/min,
    given the conditions of the problem. -/
theorem school_walk_time_difference (distance : ℝ) (slow_rate fast_rate : ℝ) (early_time : ℝ) :
  distance = 630 →
  slow_rate = 5 →
  fast_rate = 7 →
  early_time = 30 →
  distance / fast_rate + early_time = distance / slow_rate →
  distance / slow_rate - distance / fast_rate = 6 :=
by sorry

end NUMINAMATH_CALUDE_school_walk_time_difference_l2407_240760


namespace NUMINAMATH_CALUDE_derivative_of_log2_l2407_240701

-- Define the base-2 logarithm
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem derivative_of_log2 (x : ℝ) (h : x > 0) :
  deriv log2 x = 1 / (x * Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_derivative_of_log2_l2407_240701


namespace NUMINAMATH_CALUDE_total_fruits_is_137_l2407_240734

/-- The number of fruits picked by George, Amelia, and Olivia -/
def total_fruits (george_oranges : ℕ) (amelia_apples : ℕ) (amelia_orange_diff : ℕ) 
  (george_apple_diff : ℕ) (olivia_time : ℕ) (olivia_orange_rate : ℕ) (olivia_apple_rate : ℕ) 
  (olivia_time_unit : ℕ) : ℕ :=
  let george_apples := amelia_apples + george_apple_diff
  let amelia_oranges := george_oranges - amelia_orange_diff
  let olivia_cycles := olivia_time / olivia_time_unit
  let olivia_oranges := olivia_orange_rate * olivia_cycles
  let olivia_apples := olivia_apple_rate * olivia_cycles
  george_oranges + george_apples + amelia_oranges + amelia_apples + olivia_oranges + olivia_apples

/-- Theorem stating that the total number of fruits picked is 137 -/
theorem total_fruits_is_137 : 
  total_fruits 45 15 18 5 30 3 2 5 = 137 := by
  sorry


end NUMINAMATH_CALUDE_total_fruits_is_137_l2407_240734


namespace NUMINAMATH_CALUDE_expand_expression_l2407_240737

theorem expand_expression (x y : ℝ) : (x + 12) * (3 * y + 8) = 3 * x * y + 8 * x + 36 * y + 96 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2407_240737


namespace NUMINAMATH_CALUDE_total_spent_is_83_50_l2407_240736

-- Define the ticket prices
def adult_ticket_price : ℚ := 5.5
def child_ticket_price : ℚ := 3.5

-- Define the total number of tickets and number of adult tickets
def total_tickets : ℕ := 21
def adult_tickets : ℕ := 5

-- Define the function to calculate total spent
def total_spent : ℚ :=
  (adult_tickets : ℚ) * adult_ticket_price + 
  ((total_tickets - adult_tickets) : ℚ) * child_ticket_price

-- Theorem statement
theorem total_spent_is_83_50 : total_spent = 83.5 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_83_50_l2407_240736


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l2407_240711

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ m : ℕ, m < k → is_prime m → ¬(n % m = 0)

theorem smallest_non_prime_non_square_no_small_factors :
  ∀ n : ℕ, n > 0 →
    (¬is_prime n ∧ ¬is_perfect_square n ∧ has_no_prime_factor_less_than n 70) →
    n ≥ 5183 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l2407_240711
