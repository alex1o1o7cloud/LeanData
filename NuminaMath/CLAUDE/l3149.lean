import Mathlib

namespace NUMINAMATH_CALUDE_daves_shirts_l3149_314995

theorem daves_shirts (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) (not_washed : ℕ) : 
  long_sleeve = 27 →
  washed = 20 →
  not_washed = 16 →
  short_sleeve + long_sleeve = washed + not_washed →
  short_sleeve = 9 := by
sorry

end NUMINAMATH_CALUDE_daves_shirts_l3149_314995


namespace NUMINAMATH_CALUDE_gum_distribution_l3149_314975

theorem gum_distribution (cousins : ℕ) (total_gum : ℕ) (gum_per_cousin : ℕ) 
    (h1 : cousins = 4)
    (h2 : total_gum = 20)
    (h3 : total_gum = cousins * gum_per_cousin) :
  gum_per_cousin = 5 := by
  sorry

end NUMINAMATH_CALUDE_gum_distribution_l3149_314975


namespace NUMINAMATH_CALUDE_sin_120_degrees_l3149_314929

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l3149_314929


namespace NUMINAMATH_CALUDE_decreasing_implies_positive_a_l3149_314921

/-- The function f(x) = a(x^3 - 3x) is decreasing on the interval (-1, 1) --/
def is_decreasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

/-- The main theorem: if f(x) = a(x^3 - 3x) is decreasing on (-1, 1), then a > 0 --/
theorem decreasing_implies_positive_a (a : ℝ) :
  is_decreasing_on_interval (fun x => a * (x^3 - 3*x)) → a > 0 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_implies_positive_a_l3149_314921


namespace NUMINAMATH_CALUDE_max_term_of_sequence_l3149_314943

theorem max_term_of_sequence (n : ℕ) : 
  let a : ℕ → ℤ := λ k => -2 * k^2 + 9 * k + 3
  ∀ k, a k ≤ a 2 := by
  sorry

end NUMINAMATH_CALUDE_max_term_of_sequence_l3149_314943


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_16_l3149_314945

/-- A regular six-pointed star -/
structure RegularSixPointedStar :=
  (area : ℝ)
  (star_formed_by_two_triangles : Bool)
  (each_triangle_area : ℝ)

/-- The area of a quadrilateral formed by two adjacent points and the center of the star -/
def quadrilateral_area (star : RegularSixPointedStar) : ℝ := sorry

/-- Theorem stating the area of the quadrilateral is 16 cm² -/
theorem quadrilateral_area_is_16 (star : RegularSixPointedStar) 
  (h1 : star.star_formed_by_two_triangles = true) 
  (h2 : star.each_triangle_area = 72) : quadrilateral_area star = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_16_l3149_314945


namespace NUMINAMATH_CALUDE_base3_addition_theorem_l3149_314958

/-- Convert a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Convert a decimal number to its base 3 representation -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else go (m / 3) ((m % 3) :: acc)
    go n []

theorem base3_addition_theorem :
  let a := [2]
  let b := [2, 2]
  let c := [2, 0, 2]
  let d := [2, 2, 0, 2]
  let result := [0, 1, 0, 1, 2]
  base3ToDecimal a + base3ToDecimal b + base3ToDecimal c + base3ToDecimal d =
  base3ToDecimal result := by
  sorry

end NUMINAMATH_CALUDE_base3_addition_theorem_l3149_314958


namespace NUMINAMATH_CALUDE_complement_of_union_l3149_314983

/-- Given sets U, A, and B, prove that the complement of their union in U is {5} -/
theorem complement_of_union (U A B : Set ℕ) : 
  U = {1, 3, 5, 9} → A = {1, 3, 9} → B = {1, 9} → 
  (U \ (A ∪ B)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3149_314983


namespace NUMINAMATH_CALUDE_max_area_triangle_is_isosceles_l3149_314982

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Check if a point lies on a circle -/
def onCircle (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Calculate the area of a triangle given its vertices -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  sorry

/-- The theorem stating that the triangle of maximum area is isosceles -/
theorem max_area_triangle_is_isosceles
  (c : Circle) (p : Point) (h : ¬ onCircle c p) :
  ∃ (a b : Point),
    onCircle c a ∧ onCircle c b ∧
    ∀ (x y : Point),
      onCircle c x → onCircle c y →
      triangleArea p x y ≤ triangleArea p a b →
      (p.1 - a.1)^2 + (p.2 - a.2)^2 = (p.1 - b.1)^2 + (p.2 - b.2)^2 :=
sorry

end NUMINAMATH_CALUDE_max_area_triangle_is_isosceles_l3149_314982


namespace NUMINAMATH_CALUDE_six_balls_two_boxes_at_least_two_l3149_314971

/-- The number of ways to distribute n distinguishable balls into 2 distinguishable boxes -/
def totalArrangements (n : ℕ) : ℕ := 2^n

/-- The number of ways to choose k balls from n distinguishable balls -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to distribute n distinguishable balls into 2 distinguishable boxes
    where one box must contain at least m balls -/
def validArrangements (n m : ℕ) : ℕ :=
  totalArrangements n - (choose n 0 + choose n 1)

theorem six_balls_two_boxes_at_least_two :
  validArrangements 6 2 = 57 := by sorry

end NUMINAMATH_CALUDE_six_balls_two_boxes_at_least_two_l3149_314971


namespace NUMINAMATH_CALUDE_plane_equation_correct_l3149_314979

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Parametric representation of a plane -/
def parametricPlane (s t : ℝ) : Point3D :=
  { x := 2 + 2*s - t
    y := 4 - 2*s
    z := 5 - 3*s + 3*t }

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The plane equation we want to prove -/
def targetPlane : Plane :=
  { a := 2
    b := 1
    c := -1
    d := -3 }

theorem plane_equation_correct :
  ∀ s t : ℝ, pointOnPlane targetPlane (parametricPlane s t) := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l3149_314979


namespace NUMINAMATH_CALUDE_systematic_sampling_problem_l3149_314901

/-- Systematic sampling function -/
def systematicSample (start : ℕ) (interval : ℕ) (n : ℕ) : ℕ :=
  start + interval * (n - 1)

/-- Theorem for systematic sampling in the given problem -/
theorem systematic_sampling_problem :
  let totalStudents : ℕ := 500
  let selectedStudents : ℕ := 50
  let interval : ℕ := totalStudents / selectedStudents
  let start : ℕ := 6
  ∀ n : ℕ, 
    125 ≤ systematicSample start interval n ∧ 
    systematicSample start interval n ≤ 140 → 
    systematicSample start interval n = 126 ∨ 
    systematicSample start interval n = 136 :=
by
  sorry

#check systematic_sampling_problem

end NUMINAMATH_CALUDE_systematic_sampling_problem_l3149_314901


namespace NUMINAMATH_CALUDE_max_player_salary_l3149_314917

theorem max_player_salary (n : ℕ) (min_salary max_total : ℝ) :
  n = 25 →
  min_salary = 20000 →
  max_total = 900000 →
  let max_single_salary := max_total - (n - 1) * min_salary
  max_single_salary = 420000 :=
by
  sorry

end NUMINAMATH_CALUDE_max_player_salary_l3149_314917


namespace NUMINAMATH_CALUDE_ruel_stamps_l3149_314992

/-- The number of stamps in a book of 10 stamps -/
def stamps_per_book_10 : ℕ := 10

/-- The number of stamps in a book of 15 stamps -/
def stamps_per_book_15 : ℕ := 15

/-- The number of books with 10 stamps -/
def books_10 : ℕ := 4

/-- The number of books with 15 stamps -/
def books_15 : ℕ := 6

/-- The total number of stamps Ruel has -/
def total_stamps : ℕ := books_10 * stamps_per_book_10 + books_15 * stamps_per_book_15

theorem ruel_stamps : total_stamps = 130 := by
  sorry

end NUMINAMATH_CALUDE_ruel_stamps_l3149_314992


namespace NUMINAMATH_CALUDE_choir_members_count_l3149_314966

theorem choir_members_count : ∃! n : ℕ, 300 ≤ n ∧ n ≤ 400 ∧ n % 12 = 10 ∧ n % 14 = 12 := by
  sorry

end NUMINAMATH_CALUDE_choir_members_count_l3149_314966


namespace NUMINAMATH_CALUDE_subset_union_implies_complement_superset_l3149_314956

universe u

theorem subset_union_implies_complement_superset
  {U : Type u} [CompleteLattice U]
  (M N : Set U) (h : M ∪ N = N) :
  (M : Set U)ᶜ ⊇ (N : Set U)ᶜ :=
by sorry

end NUMINAMATH_CALUDE_subset_union_implies_complement_superset_l3149_314956


namespace NUMINAMATH_CALUDE_function_and_angle_theorem_l3149_314903

/-- Given a function f and an angle α, proves that f(x) = cos x and 
    (√2 f(2α - π/4) - 1) / (1 - tan α) = 2/5 under certain conditions -/
theorem function_and_angle_theorem (f : ℝ → ℝ) (ω φ α : ℝ) : 
  ω > 0 → 
  0 ≤ φ ∧ φ ≤ π → 
  (∀ x, f x = Real.sin (ω * x + φ)) →
  (∀ x, f x = f (-x)) →
  (∃ x₁ x₂, abs (x₁ - x₂) = Real.sqrt (4 + Real.pi^2) ∧ 
    f x₁ = 1 ∧ f x₂ = -1) →
  Real.tan α + 1 / Real.tan α = 5 →
  (∀ x, f x = Real.cos x) ∧ 
  (Real.sqrt 2 * f (2 * α - Real.pi / 4) - 1) / (1 - Real.tan α) = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_function_and_angle_theorem_l3149_314903


namespace NUMINAMATH_CALUDE_jasper_kite_raising_time_l3149_314936

/-- Given Omar's kite-raising rate and Jasper's relative speed, prove Jasper's time to raise his kite -/
theorem jasper_kite_raising_time
  (omar_height : ℝ)
  (omar_time : ℝ)
  (jasper_speed_ratio : ℝ)
  (jasper_height : ℝ)
  (h1 : omar_height = 240)
  (h2 : omar_time = 12)
  (h3 : jasper_speed_ratio = 3)
  (h4 : jasper_height = 600) :
  (jasper_height / (jasper_speed_ratio * (omar_height / omar_time))) = 10 :=
by sorry

end NUMINAMATH_CALUDE_jasper_kite_raising_time_l3149_314936


namespace NUMINAMATH_CALUDE_watermelon_price_in_ten_thousand_won_l3149_314981

/-- The price of a watermelon in won -/
def watermelon_price : ℝ := 50000 - 2000

/-- Conversion factor from won to ten thousand won -/
def won_to_ten_thousand : ℝ := 10000

theorem watermelon_price_in_ten_thousand_won : 
  watermelon_price / won_to_ten_thousand = 4.8 := by sorry

end NUMINAMATH_CALUDE_watermelon_price_in_ten_thousand_won_l3149_314981


namespace NUMINAMATH_CALUDE_watch_gain_percentage_l3149_314908

/-- Calculates the gain percentage when a watch is sold at a higher price -/
theorem watch_gain_percentage (cost_price : ℝ) (loss_percentage : ℝ) (price_increase : ℝ) : 
  cost_price = 1400 →
  loss_percentage = 10 →
  price_increase = 196 →
  let initial_selling_price := cost_price * (1 - loss_percentage / 100)
  let new_selling_price := initial_selling_price + price_increase
  let gain_amount := new_selling_price - cost_price
  let gain_percentage := (gain_amount / cost_price) * 100
  gain_percentage = 4 := by
  sorry

end NUMINAMATH_CALUDE_watch_gain_percentage_l3149_314908


namespace NUMINAMATH_CALUDE_escalator_travel_time_l3149_314922

/-- Calculates the time taken for a person to cover the length of a moving escalator -/
theorem escalator_travel_time 
  (escalator_speed : ℝ) 
  (escalator_length : ℝ) 
  (person_speed : ℝ) : 
  escalator_speed = 12 →
  escalator_length = 210 →
  person_speed = 2 →
  escalator_length / (escalator_speed + person_speed) = 15 := by
sorry


end NUMINAMATH_CALUDE_escalator_travel_time_l3149_314922


namespace NUMINAMATH_CALUDE_sweets_remaining_problem_l3149_314932

/-- The number of sweets remaining in a packet after some are eaten and given away -/
def sweets_remaining (cherry strawberry pineapple : ℕ) : ℕ :=
  let total := cherry + strawberry + pineapple
  let eaten := (cherry / 2) + (strawberry / 2) + (pineapple / 2)
  let given_away := 5
  total - eaten - given_away

/-- Theorem stating the number of sweets remaining in the packet -/
theorem sweets_remaining_problem :
  sweets_remaining 30 40 50 = 55 := by
  sorry

end NUMINAMATH_CALUDE_sweets_remaining_problem_l3149_314932


namespace NUMINAMATH_CALUDE_fourth_person_height_l3149_314909

theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℝ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- heights in increasing order
  h₂ - h₁ = 2 →                 -- difference between 1st and 2nd
  h₃ - h₂ = 2 →                 -- difference between 2nd and 3rd
  h₄ - h₃ = 6 →                 -- difference between 3rd and 4th
  (h₁ + h₂ + h₃ + h₄) / 4 = 76  -- average height
  → h₄ = 82 :=                  -- height of 4th person
by sorry

end NUMINAMATH_CALUDE_fourth_person_height_l3149_314909


namespace NUMINAMATH_CALUDE_inverse_81_mod_103_l3149_314970

theorem inverse_81_mod_103 (h : (9⁻¹ : ZMod 103) = 65) : (81⁻¹ : ZMod 103) = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_81_mod_103_l3149_314970


namespace NUMINAMATH_CALUDE_expression_evaluation_l3149_314947

theorem expression_evaluation : 6 * 199 + 4 * 199 + 3 * 199 + 199 + 100 = 2886 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3149_314947


namespace NUMINAMATH_CALUDE_sphere_radius_from_hole_l3149_314962

/-- Given a spherical hole in ice with a diameter of 30 cm at the surface and a depth of 10 cm,
    the radius of the sphere that created this hole is 16.25 cm. -/
theorem sphere_radius_from_hole (diameter : ℝ) (depth : ℝ) (radius : ℝ) :
  diameter = 30 ∧ depth = 10 ∧ radius = (diameter / 2)^2 / (4 * depth) + depth / 4 →
  radius = 16.25 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_from_hole_l3149_314962


namespace NUMINAMATH_CALUDE_larger_number_proof_l3149_314946

theorem larger_number_proof (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  Nat.gcd a b = 60 → Nat.lcm a b = 9900 → max a b = 900 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3149_314946


namespace NUMINAMATH_CALUDE_donnelly_class_size_l3149_314934

/-- The number of cupcakes Quinton brought to school -/
def total_cupcakes : ℕ := 40

/-- The number of students in Ms. Delmont's class -/
def delmont_students : ℕ := 18

/-- The number of staff members who received cupcakes -/
def staff_recipients : ℕ := 4

/-- The number of leftover cupcakes -/
def leftover_cupcakes : ℕ := 2

/-- The number of students in Mrs. Donnelly's class -/
def donnelly_students : ℕ := total_cupcakes - delmont_students - staff_recipients - leftover_cupcakes

theorem donnelly_class_size : donnelly_students = 16 := by
  sorry

end NUMINAMATH_CALUDE_donnelly_class_size_l3149_314934


namespace NUMINAMATH_CALUDE_two_digit_R_equal_l3149_314988

/-- R(n) is the sum of remainders when n is divided by 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, and 12 -/
def R (n : ℕ) : ℕ :=
  (n % 2) + (n % 3) + (n % 4) + (n % 5) + (n % 6) + (n % 7) + (n % 8) + (n % 9) + (n % 10) + (n % 11) + (n % 12)

/-- A two-digit positive integer is between 10 and 99, inclusive -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The theorem states that there are exactly 2 two-digit positive integers n such that R(n) = R(n+2) -/
theorem two_digit_R_equal : ∃! (s : Finset ℕ), 
  (∀ n ∈ s, is_two_digit n ∧ R n = R (n + 2)) ∧ Finset.card s = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_R_equal_l3149_314988


namespace NUMINAMATH_CALUDE_correct_calculation_l3149_314915

theorem correct_calculation (x : ℝ) (h : 6 * x = 42) : 3 * x = 21 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3149_314915


namespace NUMINAMATH_CALUDE_circle_inequality_l3149_314985

theorem circle_inequality (a b c d : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hab : a * b + c * d = 1)
  (h1 : x₁^2 + y₁^2 = 1) (h2 : x₂^2 + y₂^2 = 1)
  (h3 : x₃^2 + y₃^2 = 1) (h4 : x₄^2 + y₄^2 = 1) :
  (a * y₁ + b * y₂ + c * y₃ + d * y₄)^2 + (a * x₄ + b * x₃ + c * x₂ + d * x₁)^2 
  ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) :=
by sorry

end NUMINAMATH_CALUDE_circle_inequality_l3149_314985


namespace NUMINAMATH_CALUDE_range_of_a_l3149_314973

def f (a x : ℝ) : ℝ := x^2 - 2*(a-2)*x + a

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, (x < 1 ∨ x > 5) → f a x > 0) → a ∈ Set.Ioo 4 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3149_314973


namespace NUMINAMATH_CALUDE_count_twelve_digit_numbers_with_three_ones_l3149_314948

/-- Recursively defines the count of n-digit numbers with digits 1 or 2 without three consecutive 1's -/
def G : ℕ → ℕ
| 0 => 1  -- Base case for 0 digits (empty string)
| 1 => 2  -- Base case for 1 digit
| 2 => 3  -- Base case for 2 digits
| n + 3 => G (n + 2) + G (n + 1) + G n

/-- The count of 12-digit numbers with all digits 1 or 2 and at least three consecutive 1's -/
def count_with_three_ones : ℕ := 2^12 - G 12

theorem count_twelve_digit_numbers_with_three_ones : 
  count_with_three_ones = 3656 :=
sorry

end NUMINAMATH_CALUDE_count_twelve_digit_numbers_with_three_ones_l3149_314948


namespace NUMINAMATH_CALUDE_egyptian_fraction_equation_solutions_l3149_314996

theorem egyptian_fraction_equation_solutions :
  ∀ x y z : ℕ+,
  (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = 4 / 5 →
  ((x = 2 ∧ y = 4 ∧ z = 20) ∨ (x = 2 ∧ y = 5 ∧ z = 10)) :=
by sorry

end NUMINAMATH_CALUDE_egyptian_fraction_equation_solutions_l3149_314996


namespace NUMINAMATH_CALUDE_cow_herd_distribution_l3149_314950

theorem cow_herd_distribution (total : ℕ) : 
  (total : ℚ) / 3 + (total : ℚ) / 6 + (total : ℚ) / 8 + 9 = total → total = 216 := by
  sorry

end NUMINAMATH_CALUDE_cow_herd_distribution_l3149_314950


namespace NUMINAMATH_CALUDE_ellipse_parameter_sum_l3149_314951

-- Define the foci
def F₁ : ℝ × ℝ := (0, 2)
def F₂ : ℝ × ℝ := (8, 2)

-- Define the ellipse
def Ellipse : Set (ℝ × ℝ) :=
  {P | Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
       Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 12}

-- Define the ellipse equation parameters
noncomputable def h : ℝ := (F₁.1 + F₂.1) / 2
noncomputable def k : ℝ := (F₁.2 + F₂.2) / 2
noncomputable def a : ℝ := 6
noncomputable def b : ℝ := Real.sqrt (a^2 - ((F₂.1 - F₁.1) / 2)^2)

-- Theorem statement
theorem ellipse_parameter_sum :
  h + k + a + b = 12 + 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_parameter_sum_l3149_314951


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3149_314902

/-- Given a complex number z satisfying (1 + z) / i = 1 - z, 
    the imaginary part of z is 1 -/
theorem imaginary_part_of_z (z : ℂ) (h : (1 + z) / Complex.I = 1 - z) : 
  Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3149_314902


namespace NUMINAMATH_CALUDE_limit_sin_difference_l3149_314984

theorem limit_sin_difference (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ →
    |(1 / (4 * Real.sin x ^ 2) - 1 / Real.sin (2 * x) ^ 2) - (-1/4)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_sin_difference_l3149_314984


namespace NUMINAMATH_CALUDE_total_suitcases_l3149_314959

/-- The number of siblings in Lily's family -/
def num_siblings : Nat := 6

/-- The number of parents in Lily's family -/
def num_parents : Nat := 2

/-- The number of grandparents in Lily's family -/
def num_grandparents : Nat := 2

/-- The number of other relatives in Lily's family -/
def num_other_relatives : Nat := 3

/-- The number of suitcases each parent brings -/
def suitcases_per_parent : Nat := 3

/-- The number of suitcases each grandparent brings -/
def suitcases_per_grandparent : Nat := 2

/-- The total number of suitcases brought by other relatives -/
def suitcases_other_relatives : Nat := 8

/-- The sum of suitcases brought by siblings -/
def siblings_suitcases : Nat := (List.range num_siblings).sum.succ

/-- The total number of suitcases brought by Lily's family -/
theorem total_suitcases : 
  siblings_suitcases + 
  (num_parents * suitcases_per_parent) + 
  (num_grandparents * suitcases_per_grandparent) + 
  suitcases_other_relatives = 39 := by
  sorry

end NUMINAMATH_CALUDE_total_suitcases_l3149_314959


namespace NUMINAMATH_CALUDE_quadratic_form_h_value_l3149_314913

theorem quadratic_form_h_value :
  ∃ (a k : ℝ), ∀ x, 3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_h_value_l3149_314913


namespace NUMINAMATH_CALUDE_c_investment_approx_l3149_314930

/-- Calculates the investment of partner C in a partnership business --/
def calculate_c_investment (a_investment b_investment total_profit a_profit : ℚ) : ℚ :=
  let total_investment := a_investment + b_investment + (12100 * a_investment / a_profit - a_investment - b_investment)
  12100 * a_investment / a_profit - a_investment - b_investment

/-- Theorem stating that C's investment is approximately 10492 --/
theorem c_investment_approx (a_investment b_investment total_profit a_profit : ℚ) 
  (h1 : a_investment = 6300)
  (h2 : b_investment = 4200)
  (h3 : total_profit = 12100)
  (h4 : a_profit = 3630) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  abs (calculate_c_investment a_investment b_investment total_profit a_profit - 10492) < ε :=
sorry

end NUMINAMATH_CALUDE_c_investment_approx_l3149_314930


namespace NUMINAMATH_CALUDE_even_function_implies_b_zero_l3149_314963

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = x(x+b) -/
def f (b : ℝ) : ℝ → ℝ := λ x ↦ x * (x + b)

/-- If f(x) = x(x+b) is an even function, then b = 0 -/
theorem even_function_implies_b_zero :
  ∀ b : ℝ, IsEven (f b) → b = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_b_zero_l3149_314963


namespace NUMINAMATH_CALUDE_triangle_properties_l3149_314987

/-- Given a triangle with sides 8, 15, and 17, prove it's a right triangle
    and find the longest side of a similar triangle with perimeter 160 -/
theorem triangle_properties (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  (a^2 + b^2 = c^2) ∧ 
  (∃ (x : ℝ), x * (a + b + c) = 160 ∧ x * c = 68) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3149_314987


namespace NUMINAMATH_CALUDE_perfect_squares_as_sum_of_odd_composites_l3149_314997

def is_odd_composite (n : ℕ) : Prop := n % 2 = 1 ∧ ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b

def is_sum_of_three_odd_composites (n : ℕ) : Prop :=
  ∃ a b c, is_odd_composite a ∧ is_odd_composite b ∧ is_odd_composite c ∧ n = a + b + c

def perfect_square_set : Set ℕ := {n | ∃ k : ℕ, k ≥ 3 ∧ n = (2 * k + 1)^2}

theorem perfect_squares_as_sum_of_odd_composites :
  ∀ n : ℕ, n ∈ perfect_square_set ↔ is_sum_of_three_odd_composites n :=
sorry

end NUMINAMATH_CALUDE_perfect_squares_as_sum_of_odd_composites_l3149_314997


namespace NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l3149_314969

theorem permutations_of_six_distinct_objects : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l3149_314969


namespace NUMINAMATH_CALUDE_exact_sunny_days_probability_l3149_314980

def num_days : ℕ := 5
def sunny_prob : ℚ := 2/5
def desired_sunny_days : ℕ := 2

theorem exact_sunny_days_probability :
  (num_days.choose desired_sunny_days : ℚ) * sunny_prob ^ desired_sunny_days * (1 - sunny_prob) ^ (num_days - desired_sunny_days) = 4320/15625 := by
  sorry

end NUMINAMATH_CALUDE_exact_sunny_days_probability_l3149_314980


namespace NUMINAMATH_CALUDE_xy_value_l3149_314904

theorem xy_value (x y : ℝ) (h : x^2 + y^2 - 22*x - 20*y + 221 = 0) : x * y = 110 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3149_314904


namespace NUMINAMATH_CALUDE_three_hundredth_term_omit_squares_l3149_314972

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def omit_squares_sequence (n : ℕ) : ℕ :=
  n + (Nat.sqrt n)

theorem three_hundredth_term_omit_squares : omit_squares_sequence 300 = 317 := by
  sorry

end NUMINAMATH_CALUDE_three_hundredth_term_omit_squares_l3149_314972


namespace NUMINAMATH_CALUDE_olivia_picked_16_pieces_l3149_314986

/-- The number of pieces of paper Olivia picked up -/
def olivia_pieces : ℕ := 19 - 3

/-- The number of pieces of paper Edward picked up -/
def edward_pieces : ℕ := 3

/-- The total number of pieces of paper picked up -/
def total_pieces : ℕ := 19

theorem olivia_picked_16_pieces :
  olivia_pieces = 16 ∧ olivia_pieces + edward_pieces = total_pieces :=
sorry

end NUMINAMATH_CALUDE_olivia_picked_16_pieces_l3149_314986


namespace NUMINAMATH_CALUDE_car_tire_usage_l3149_314978

/-- Represents the usage of tires on a car -/
structure TireUsage where
  total_tires : ℕ
  active_tires : ℕ
  total_miles : ℕ
  miles_per_tire : ℕ

/-- Calculates the miles each tire is used given the total miles driven and number of tires -/
def calculate_miles_per_tire (usage : TireUsage) : Prop :=
  usage.miles_per_tire = usage.total_miles * usage.active_tires / usage.total_tires

/-- Theorem stating that for a car with 5 tires, 4 of which are used at any time, 
    each tire is used for 40,000 miles over a total of 50,000 miles driven -/
theorem car_tire_usage :
  ∀ (usage : TireUsage), 
    usage.total_tires = 5 →
    usage.active_tires = 4 →
    usage.total_miles = 50000 →
    calculate_miles_per_tire usage →
    usage.miles_per_tire = 40000 :=
sorry

end NUMINAMATH_CALUDE_car_tire_usage_l3149_314978


namespace NUMINAMATH_CALUDE_student_count_l3149_314968

theorem student_count (n : ℕ) (ella : ℕ) : 
  (ella = 60) → -- Ella's position from best
  (n + 1 - ella = 60) → -- Ella's position from worst (n is total students minus 1)
  (n + 1 = 119) := by
sorry

end NUMINAMATH_CALUDE_student_count_l3149_314968


namespace NUMINAMATH_CALUDE_smallest_perimeter_is_78_l3149_314923

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the point J (intersection of angle bisectors)
def J : ℝ × ℝ := sorry

-- Define the condition that PQR has positive integer side lengths
def has_positive_integer_sides (t : Triangle) : Prop :=
  ∃ (a b c : ℕ+), 
    dist t.P t.Q = a ∧ 
    dist t.Q t.R = b ∧ 
    dist t.R t.P = c

-- Define the condition that PQR is isosceles with PQ = PR
def is_isosceles (t : Triangle) : Prop :=
  dist t.P t.Q = dist t.P t.R

-- Define the condition that J is on the angle bisectors of ∠Q and ∠R
def J_on_angle_bisectors (t : Triangle) : Prop :=
  sorry

-- Define the condition that QJ = 10
def QJ_equals_10 (t : Triangle) : Prop :=
  dist t.Q J = 10

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ :=
  dist t.P t.Q + dist t.Q t.R + dist t.R t.P

-- Theorem statement
theorem smallest_perimeter_is_78 :
  ∀ t : Triangle,
    has_positive_integer_sides t →
    is_isosceles t →
    J_on_angle_bisectors t →
    QJ_equals_10 t →
    ∀ t' : Triangle,
      has_positive_integer_sides t' →
      is_isosceles t' →
      J_on_angle_bisectors t' →
      QJ_equals_10 t' →
      perimeter t ≤ perimeter t' →
      perimeter t = 78 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_is_78_l3149_314923


namespace NUMINAMATH_CALUDE_max_sum_with_product_2665_l3149_314938

theorem max_sum_with_product_2665 :
  ∀ A B C : ℕ+,
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2665 →
  A + B + C ≤ 539 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_product_2665_l3149_314938


namespace NUMINAMATH_CALUDE_nancy_sweaters_count_l3149_314916

/-- Represents the washing machine capacity -/
def machine_capacity : ℕ := 9

/-- Represents the number of shirts Nancy had to wash -/
def number_of_shirts : ℕ := 19

/-- Represents the total number of loads Nancy did -/
def total_loads : ℕ := 3

/-- Calculates the number of sweaters Nancy had to wash -/
def number_of_sweaters : ℕ := machine_capacity

theorem nancy_sweaters_count :
  number_of_sweaters = machine_capacity := by sorry

end NUMINAMATH_CALUDE_nancy_sweaters_count_l3149_314916


namespace NUMINAMATH_CALUDE_area_of_specific_rectangle_l3149_314940

/-- Represents a rectangle with given properties -/
structure Rectangle where
  breadth : ℝ
  length : ℝ
  perimeter : ℝ
  area : ℝ

/-- Theorem: Area of a specific rectangle -/
theorem area_of_specific_rectangle :
  ∀ (rect : Rectangle),
  rect.length = 3 * rect.breadth →
  rect.perimeter = 104 →
  rect.area = rect.length * rect.breadth →
  rect.area = 507 := by
sorry

end NUMINAMATH_CALUDE_area_of_specific_rectangle_l3149_314940


namespace NUMINAMATH_CALUDE_division_theorem_l3149_314919

theorem division_theorem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 140 → 
  divisor = 15 → 
  remainder = 5 → 
  dividend = divisor * quotient + remainder → 
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l3149_314919


namespace NUMINAMATH_CALUDE_cyclists_distance_l3149_314939

/-- Calculates the distance between two cyclists traveling in opposite directions -/
def distance_between_cyclists (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 + speed2) * time

theorem cyclists_distance :
  let speed1 : ℝ := 10
  let speed2 : ℝ := 25
  let time : ℝ := 1.4285714285714286
  distance_between_cyclists speed1 speed2 time = 50 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_distance_l3149_314939


namespace NUMINAMATH_CALUDE_pipe_filling_time_l3149_314999

theorem pipe_filling_time (p q r : ℝ) (hp : p = 3) (hr : r = 18) (hall : 1/p + 1/q + 1/r = 1/2) :
  q = 9 := by
sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l3149_314999


namespace NUMINAMATH_CALUDE_cyclic_quadrilaterals_count_l3149_314960

/-- A quadrilateral is cyclic if it can be inscribed in a circle. -/
def is_cyclic (q : Quadrilateral) : Prop := sorry

/-- A square is a quadrilateral with all sides equal and all angles right angles. -/
def is_square (q : Quadrilateral) : Prop := sorry

/-- A rectangle is a quadrilateral with all angles right angles. -/
def is_rectangle (q : Quadrilateral) : Prop := sorry

/-- A rhombus is a quadrilateral with all sides equal. -/
def is_rhombus (q : Quadrilateral) : Prop := sorry

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
def is_parallelogram (q : Quadrilateral) : Prop := sorry

/-- An isosceles trapezoid is a trapezoid with the non-parallel sides equal. -/
def is_isosceles_trapezoid (q : Quadrilateral) : Prop := sorry

theorem cyclic_quadrilaterals_count :
  ∃ (s r h p t : Quadrilateral),
    is_square s ∧
    is_rectangle r ∧ ¬ is_square r ∧
    is_rhombus h ∧ ¬ is_square h ∧
    is_parallelogram p ∧ ¬ is_rectangle p ∧ ¬ is_rhombus p ∧
    is_isosceles_trapezoid t ∧ ¬ is_parallelogram t ∧
    (is_cyclic s ∧ is_cyclic r ∧ is_cyclic t ∧
     ¬ is_cyclic h ∧ ¬ is_cyclic p) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilaterals_count_l3149_314960


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3149_314955

theorem negation_of_universal_proposition (a b : ℝ) :
  ¬(a < b → ∀ c : ℝ, a * c^2 < b * c^2) ↔ (a < b → ∃ c : ℝ, a * c^2 ≥ b * c^2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3149_314955


namespace NUMINAMATH_CALUDE_division_37_by_8_l3149_314924

theorem division_37_by_8 (A B : ℕ) : 37 = 8 * A + B ∧ B < 8 → A = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_37_by_8_l3149_314924


namespace NUMINAMATH_CALUDE_bike_ride_problem_l3149_314952

/-- Bike ride problem -/
theorem bike_ride_problem (total_distance : ℝ) (total_time : ℝ) (rest_time : ℝ) 
  (fast_speed : ℝ) (slow_speed : ℝ) 
  (h1 : total_distance = 142)
  (h2 : total_time = 8)
  (h3 : rest_time = 0.5)
  (h4 : fast_speed = 22)
  (h5 : slow_speed = 15) :
  ∃ energetic_time : ℝ, 
    energetic_time * fast_speed + (total_time - rest_time - energetic_time) * slow_speed = total_distance ∧ 
    energetic_time = 59 / 14 := by
  sorry


end NUMINAMATH_CALUDE_bike_ride_problem_l3149_314952


namespace NUMINAMATH_CALUDE_cafeteria_pies_l3149_314911

/-- Given a cafeteria with initial apples, apples handed out, and apples per pie,
    calculate the number of pies that can be made. -/
def calculate_pies (initial_apples : ℕ) (apples_handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - apples_handed_out) / apples_per_pie

/-- Theorem stating that with 96 initial apples, 42 apples handed out,
    and 6 apples per pie, the cafeteria can make 9 pies. -/
theorem cafeteria_pies :
  calculate_pies 96 42 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l3149_314911


namespace NUMINAMATH_CALUDE_nutty_professor_mixture_l3149_314977

/-- The Nutty Professor's nut mixture problem -/
theorem nutty_professor_mixture
  (cashew_price : ℝ)
  (brazil_price : ℝ)
  (mixture_price : ℝ)
  (cashew_weight : ℝ)
  (h1 : cashew_price = 6.75)
  (h2 : brazil_price = 5.00)
  (h3 : mixture_price = 5.70)
  (h4 : cashew_weight = 20)
  : ∃ (brazil_weight : ℝ),
    cashew_weight * cashew_price + brazil_weight * brazil_price =
    (cashew_weight + brazil_weight) * mixture_price ∧
    cashew_weight + brazil_weight = 50 :=
by sorry

end NUMINAMATH_CALUDE_nutty_professor_mixture_l3149_314977


namespace NUMINAMATH_CALUDE_not_right_triangle_A_right_triangle_B_right_triangle_C_right_triangle_D_main_result_l3149_314926

/-- A function to check if three numbers can form a right triangle --/
def isRightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

/-- Theorem stating that √3, 2, √5 cannot form a right triangle --/
theorem not_right_triangle_A : ¬ isRightTriangle (Real.sqrt 3) 2 (Real.sqrt 5) := by
  sorry

/-- Theorem stating that 3, 4, 5 can form a right triangle --/
theorem right_triangle_B : isRightTriangle 3 4 5 := by
  sorry

/-- Theorem stating that 0.6, 0.8, 1 can form a right triangle --/
theorem right_triangle_C : isRightTriangle 0.6 0.8 1 := by
  sorry

/-- Theorem stating that 130, 120, 50 can form a right triangle --/
theorem right_triangle_D : isRightTriangle 130 120 50 := by
  sorry

/-- Main theorem combining all the above results --/
theorem main_result : 
  ¬ isRightTriangle (Real.sqrt 3) 2 (Real.sqrt 5) ∧
  isRightTriangle 3 4 5 ∧
  isRightTriangle 0.6 0.8 1 ∧
  isRightTriangle 130 120 50 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_A_right_triangle_B_right_triangle_C_right_triangle_D_main_result_l3149_314926


namespace NUMINAMATH_CALUDE_count_multiples_of_12_and_9_l3149_314964

def count_multiples (lower upper divisor : ℕ) : ℕ :=
  (upper / divisor) - ((lower - 1) / divisor)

theorem count_multiples_of_12_and_9 : 
  count_multiples 50 400 (Nat.lcm 12 9) = 10 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_12_and_9_l3149_314964


namespace NUMINAMATH_CALUDE_fifth_month_sale_l3149_314927

def sale_month1 : ℕ := 6535
def sale_month2 : ℕ := 6927
def sale_month3 : ℕ := 6855
def sale_month4 : ℕ := 7230
def sale_month6 : ℕ := 4891
def average_sale : ℕ := 6500
def num_months : ℕ := 6

theorem fifth_month_sale :
  ∃ (sale_month5 : ℕ),
    sale_month5 = average_sale * num_months - (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month6) ∧
    sale_month5 = 6562 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l3149_314927


namespace NUMINAMATH_CALUDE_paper_clip_cost_l3149_314914

/-- The cost of one box of paper clips and one package of index cards satisfying given conditions -/
def paper_clip_and_index_card_cost (p i : ℝ) : Prop :=
  15 * p + 7 * i = 55.40 ∧ 12 * p + 10 * i = 61.70

/-- The theorem stating that the cost of one box of paper clips is 1.835 -/
theorem paper_clip_cost : ∃ (p i : ℝ), paper_clip_and_index_card_cost p i ∧ p = 1.835 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_cost_l3149_314914


namespace NUMINAMATH_CALUDE_calculate_english_marks_l3149_314953

/-- Proves that given a student's marks in 4 subjects and an average across 5 subjects,
    we can determine the marks in the fifth subject. -/
theorem calculate_english_marks (math physics chem bio : ℕ) (average : ℚ)
    (h_math : math = 65)
    (h_physics : physics = 82)
    (h_chem : chem = 67)
    (h_bio : bio = 85)
    (h_average : average = 79)
    : ∃ english : ℕ, english = 96 ∧ 
      (english + math + physics + chem + bio : ℚ) / 5 = average :=
by
  sorry

end NUMINAMATH_CALUDE_calculate_english_marks_l3149_314953


namespace NUMINAMATH_CALUDE_compound_oxygen_atoms_l3149_314900

/-- Represents the atomic weights of elements in atomic mass units (amu) -/
structure AtomicWeight where
  Cu : ℝ
  C : ℝ
  O : ℝ

/-- Represents a compound with Cu, C, and O atoms -/
structure Compound where
  Cu : ℕ
  C : ℕ
  O : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (w : AtomicWeight) : ℝ :=
  c.Cu * w.Cu + c.C * w.C + c.O * w.O

/-- Theorem stating that a compound with 1 Cu, 1 C, and n O atoms
    with a molecular weight of 124 amu has 3 O atoms -/
theorem compound_oxygen_atoms
  (w : AtomicWeight)
  (h1 : w.Cu = 63.55)
  (h2 : w.C = 12.01)
  (h3 : w.O = 16.00)
  (c : Compound)
  (h4 : c.Cu = 1)
  (h5 : c.C = 1)
  (h6 : molecularWeight c w = 124) :
  c.O = 3 := by
  sorry

end NUMINAMATH_CALUDE_compound_oxygen_atoms_l3149_314900


namespace NUMINAMATH_CALUDE_work_hours_theorem_l3149_314991

def amber_hours : ℕ := 12

def armand_hours : ℕ := amber_hours / 3

def ella_hours : ℕ := 2 * amber_hours

def total_hours : ℕ := amber_hours + armand_hours + ella_hours

theorem work_hours_theorem : total_hours = 40 := by
  sorry

end NUMINAMATH_CALUDE_work_hours_theorem_l3149_314991


namespace NUMINAMATH_CALUDE_max_value_fraction_l3149_314949

theorem max_value_fraction (x : ℝ) : 
  (4 * x^2 + 12 * x + 19) / (4 * x^2 + 12 * x + 9) ≤ 11 ∧ 
  ∀ ε > 0, ∃ y : ℝ, (4 * y^2 + 12 * y + 19) / (4 * y^2 + 12 * y + 9) > 11 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3149_314949


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l3149_314910

theorem pizza_toppings_combinations : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l3149_314910


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3149_314954

theorem sin_2alpha_value (α : Real) (h1 : α ∈ (Set.Ioo 0 Real.pi)) (h2 : Real.tan (Real.pi / 4 - α) = 1 / 3) : 
  Real.sin (2 * α) = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3149_314954


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3149_314967

-- Define the conditions
def condition_p (x : ℝ) : Prop := x^2 < x
def condition_q (x : ℝ) : Prop := 1/x > 2

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, condition_q x → condition_p x) ∧
  (∃ x : ℝ, condition_p x ∧ ¬condition_q x) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3149_314967


namespace NUMINAMATH_CALUDE_lcm_of_9_12_15_l3149_314937

theorem lcm_of_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_15_l3149_314937


namespace NUMINAMATH_CALUDE_luke_total_points_l3149_314944

/-- 
Given that Luke gained 11 points in each round and played 14 rounds,
prove that the total points he scored is 154.
-/
theorem luke_total_points : 
  let points_per_round : ℕ := 11
  let number_of_rounds : ℕ := 14
  points_per_round * number_of_rounds = 154 := by sorry

end NUMINAMATH_CALUDE_luke_total_points_l3149_314944


namespace NUMINAMATH_CALUDE_phone_bill_minutes_l3149_314957

def monthly_fee : ℚ := 2
def per_minute_rate : ℚ := 12 / 100
def total_bill : ℚ := 2336 / 100

theorem phone_bill_minutes : 
  ∃ (minutes : ℕ), 
    (monthly_fee + per_minute_rate * minutes) = total_bill ∧ 
    minutes = 178 := by
  sorry

end NUMINAMATH_CALUDE_phone_bill_minutes_l3149_314957


namespace NUMINAMATH_CALUDE_division_problem_l3149_314989

theorem division_problem : (0.25 / 0.005) / 0.1 = 500 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3149_314989


namespace NUMINAMATH_CALUDE_solution_set_abs_x_times_x_minus_one_l3149_314998

theorem solution_set_abs_x_times_x_minus_one (x : ℝ) :
  (|x| * (x - 1) ≥ 0) ↔ (x ≥ 1 ∨ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_abs_x_times_x_minus_one_l3149_314998


namespace NUMINAMATH_CALUDE_fast_food_fries_l3149_314965

theorem fast_food_fries (total : ℕ) (ratio : ℚ) (small : ℕ) : 
  total = 24 → ratio = 5 → small * (1 + ratio) = total → small = 4 := by
  sorry

end NUMINAMATH_CALUDE_fast_food_fries_l3149_314965


namespace NUMINAMATH_CALUDE_intersection_and_parallel_line_equation_l3149_314994

/-- Given two lines in the plane and their intersection point, prove that a third line
    passing through the intersection point and parallel to a fourth line has a specific equation. -/
theorem intersection_and_parallel_line_equation :
  -- Define the first line: 2x - 3y - 3 = 0
  let l₁ : Set (ℝ × ℝ) := {p | 2 * p.1 - 3 * p.2 - 3 = 0}
  -- Define the second line: x + y + 2 = 0
  let l₂ : Set (ℝ × ℝ) := {p | p.1 + p.2 + 2 = 0}
  -- Define the parallel line: 3x + y - 1 = 0
  let l_parallel : Set (ℝ × ℝ) := {p | 3 * p.1 + p.2 - 1 = 0}
  -- Define the intersection point of l₁ and l₂
  let intersection : ℝ × ℝ := (-3/5, -7/5)
  -- Assume the intersection point lies on both l₁ and l₂
  (intersection ∈ l₁) ∧ (intersection ∈ l₂) →
  -- Define the line we want to prove
  let l : Set (ℝ × ℝ) := {p | 15 * p.1 + 5 * p.2 + 16 = 0}
  -- The line l passes through the intersection point
  (intersection ∈ l) ∧
  -- The line l is parallel to l_parallel
  (∀ (p q : ℝ × ℝ), p ∈ l → q ∈ l → p ≠ q →
    ∃ (r s : ℝ × ℝ), r ∈ l_parallel ∧ s ∈ l_parallel ∧ r ≠ s ∧
      (s.2 - r.2) / (s.1 - r.1) = (q.2 - p.2) / (q.1 - p.1)) :=
by
  sorry


end NUMINAMATH_CALUDE_intersection_and_parallel_line_equation_l3149_314994


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l3149_314912

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → 2*a + 8*b - a*b = 0 → x*y ≤ a*b) ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → 2*a + 8*b - a*b = 0 → x + y ≤ a + b) ∧
  x*y = 64 ∧ x + y = 18 := by
sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l3149_314912


namespace NUMINAMATH_CALUDE_meeting_time_and_bridge_location_l3149_314961

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a journey between two villages -/
structure Journey where
  startTime : TimeOfDay
  endTime : TimeOfDay
  deriving Repr

/-- Calculates the duration of a journey in minutes -/
def journeyDuration (j : Journey) : Nat :=
  (j.endTime.hours - j.startTime.hours) * 60 + j.endTime.minutes - j.startTime.minutes

/-- Theorem: Meeting time and bridge location -/
theorem meeting_time_and_bridge_location
  (womanJourney : Journey)
  (manJourney : Journey)
  (hWoman : womanJourney = ⟨⟨10, 31⟩, ⟨13, 43⟩⟩)
  (hMan : manJourney = ⟨⟨9, 13⟩, ⟨11, 53⟩⟩)
  (hSameRoad : True)  -- They travel on the same road
  (hConstantSpeed : True)  -- Both travel at constant speeds
  (hBridgeCrossing : True)  -- Woman crosses bridge 1 minute later than man
  : ∃ (meetingTime : TimeOfDay) (bridgeFromA bridgeFromB : Nat),
    meetingTime = ⟨11, 13⟩ ∧
    bridgeFromA = 7 ∧
    bridgeFromB = 24 :=
by sorry

end NUMINAMATH_CALUDE_meeting_time_and_bridge_location_l3149_314961


namespace NUMINAMATH_CALUDE_triangle_height_proof_l3149_314935

/-- Triangle ABC with vertices A(-2,10), B(2,0), and C(10,0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- A vertical line intersecting AC at R and BC at S -/
structure IntersectingLine :=
  (R : ℝ × ℝ)
  (S : ℝ × ℝ)

/-- The problem statement -/
theorem triangle_height_proof 
  (ABC : Triangle)
  (RS : IntersectingLine)
  (h1 : ABC.A = (-2, 10))
  (h2 : ABC.B = (2, 0))
  (h3 : ABC.C = (10, 0))
  (h4 : RS.R.1 = RS.S.1)  -- R and S have the same x-coordinate (vertical line)
  (h5 : RS.S.2 = 0)  -- S lies on BC (y-coordinate is 0)
  (h6 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ RS.R = (1 - t) • ABC.A + t • ABC.C)  -- R lies on AC
  (h7 : (1/2) * |RS.R.2| * 8 = 24)  -- Area of RSC is 24
  : RS.R.2 = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_height_proof_l3149_314935


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3149_314907

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (a 2 + a 6) / 2 = 5 ∧
  (a 3 + a 7) / 2 = 7

/-- The general term of the arithmetic sequence -/
def GeneralTerm (n : ℕ) : ℝ := 2 * n - 3

theorem arithmetic_sequence_general_term (a : ℕ → ℝ) :
  ArithmeticSequence a → ∀ n : ℕ, a n = GeneralTerm n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3149_314907


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3149_314974

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) → 
  (-1 < k ∧ k < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3149_314974


namespace NUMINAMATH_CALUDE_prize_orders_count_l3149_314918

/-- Represents a bowling tournament with 6 players -/
structure BowlingTournament where
  players : Fin 6

/-- Represents the number of possible outcomes for a single match -/
def match_outcomes : Nat := 2

/-- Represents the number of matches in the tournament -/
def num_matches : Nat := 5

/-- Calculates the total number of possible prize orders -/
def total_outcomes (t : BowlingTournament) : Nat :=
  match_outcomes ^ num_matches

/-- Theorem: The number of possible prize orders is 32 -/
theorem prize_orders_count (t : BowlingTournament) : 
  total_outcomes t = 32 := by
  sorry

end NUMINAMATH_CALUDE_prize_orders_count_l3149_314918


namespace NUMINAMATH_CALUDE_symmetry_about_x_equals_one_l3149_314990

theorem symmetry_about_x_equals_one (f : ℝ → ℝ) (x : ℝ) : f (x - 1) = f (-(x - 1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_about_x_equals_one_l3149_314990


namespace NUMINAMATH_CALUDE_permutation_order_l3149_314905

/-- The alphabet in its natural order -/
def T₀ : String := "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

/-- The result of applying the permutation to T₀ -/
def T₁ : String := "JQOWIPANTZRCVMYEGSHUFDKBLX"

/-- The result of applying the permutation to T₁ -/
def T₂ : String := "ZGYKTEJMUXSODVLIAHNFPWRQCB"

/-- The permutation function -/
def permutation (s : String) : String :=
  if s = T₀ then T₁
  else if s = T₁ then T₂
  else T₀  -- This else case is not explicitly given in the problem, but needed for completeness

/-- The theorem stating that the order of the permutation is 24 -/
theorem permutation_order :
  ∃ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), k > 0 ∧ k < n → (permutation^[k] T₀ ≠ T₀)) ∧ permutation^[n] T₀ = T₀ ∧ n = 24 := by
  sorry

#check permutation_order

end NUMINAMATH_CALUDE_permutation_order_l3149_314905


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3149_314933

theorem isosceles_triangle_perimeter (equilateral_perimeter : ℝ) (isosceles_base : ℝ) : 
  equilateral_perimeter = 45 → 
  isosceles_base = 10 → 
  ∃ (isosceles_side : ℝ), 
    isosceles_side = equilateral_perimeter / 3 ∧ 
    2 * isosceles_side + isosceles_base = 40 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3149_314933


namespace NUMINAMATH_CALUDE_zigzag_angle_theorem_l3149_314928

/-- A structure representing a zigzag line in a rectangle --/
structure ZigzagRectangle where
  ACB : ℝ
  FEG : ℝ
  DCE : ℝ
  DEC : ℝ

/-- The theorem stating that given specific angle measurements in a zigzag rectangle,
    the angle θ formed by the zigzag line is equal to 11 degrees --/
theorem zigzag_angle_theorem (z : ZigzagRectangle) 
  (h1 : z.ACB = 10)
  (h2 : z.FEG = 26)
  (h3 : z.DCE = 14)
  (h4 : z.DEC = 33) :
  ∃ θ : ℝ, θ = 11 := by
  sorry

end NUMINAMATH_CALUDE_zigzag_angle_theorem_l3149_314928


namespace NUMINAMATH_CALUDE_triangle_side_length_l3149_314931

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 2 + 1/2

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : f A = 1/2) 
  (h2 : a = Real.sqrt 17) 
  (h3 : b = 4) 
  (h4 : a = b * Real.sin C / Real.sin A) 
  (h5 : b = c * Real.sin A / Real.sin B) 
  (h6 : c = a * Real.sin B / Real.sin C) : 
  c = 2 + Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3149_314931


namespace NUMINAMATH_CALUDE_equation_solution_l3149_314920

theorem equation_solution : 
  ∃! x : ℝ, x ≠ 1 ∧ (2 * x) / (x - 1) - 1 = 4 / (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3149_314920


namespace NUMINAMATH_CALUDE_sphere_volume_area_ratio_l3149_314925

theorem sphere_volume_area_ratio (r R : ℝ) (h : r > 0) (H : R > 0) :
  (4 / 3 * Real.pi * r^3) / (4 / 3 * Real.pi * R^3) = 1 / 8 →
  (4 * Real.pi * r^2) / (4 * Real.pi * R^2) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_area_ratio_l3149_314925


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3149_314906

theorem min_value_of_expression (x : ℝ) (h : x > 2) : 
  4 / (x - 2) + 4 * x ≥ 16 ∧ ∃ y > 2, 4 / (y - 2) + 4 * y = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3149_314906


namespace NUMINAMATH_CALUDE_boys_without_calculators_l3149_314976

/-- Proves the number of boys without calculators in Mrs. Allen's class -/
theorem boys_without_calculators 
  (total_boys : ℕ) 
  (total_with_calculators : ℕ) 
  (girls_with_calculators : ℕ) 
  (h1 : total_boys = 20)
  (h2 : total_with_calculators = 25)
  (h3 : girls_with_calculators = 15) :
  total_boys - (total_with_calculators - girls_with_calculators) = 10 := by
  sorry

end NUMINAMATH_CALUDE_boys_without_calculators_l3149_314976


namespace NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l3149_314941

theorem quadratic_inequality_no_solution : 
  {x : ℝ | x^2 + 4*x + 4 < 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l3149_314941


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l3149_314942

theorem shaded_region_perimeter (r : ℝ) (h : r = 7) : 
  2 * r + 3 * π * r / 2 = 14 + 10.5 * π := by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l3149_314942


namespace NUMINAMATH_CALUDE_cubic_equation_property_l3149_314993

theorem cubic_equation_property (p q : ℝ) : 
  (p * 3^3 + q * 3 + 1 = 2018) → (p * (-3)^3 + q * (-3) + 1 = -2016) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_property_l3149_314993
