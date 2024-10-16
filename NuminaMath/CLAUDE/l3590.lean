import Mathlib

namespace NUMINAMATH_CALUDE_square_difference_l3590_359096

theorem square_difference (x y : ℚ) 
  (h1 : x + y = 9/17) 
  (h2 : x - y = 1/119) : 
  x^2 - y^2 = 9/2003 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3590_359096


namespace NUMINAMATH_CALUDE_fish_sales_revenue_l3590_359031

theorem fish_sales_revenue : 
  let first_week_quantity : ℕ := 50
  let first_week_price : ℚ := 10
  let second_week_quantity_multiplier : ℕ := 3
  let second_week_discount_percentage : ℚ := 25 / 100

  let first_week_revenue := first_week_quantity * first_week_price
  let second_week_quantity := first_week_quantity * second_week_quantity_multiplier
  let second_week_price := first_week_price * (1 - second_week_discount_percentage)
  let second_week_revenue := second_week_quantity * second_week_price
  let total_revenue := first_week_revenue + second_week_revenue

  total_revenue = 1625 := by
sorry

end NUMINAMATH_CALUDE_fish_sales_revenue_l3590_359031


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l3590_359014

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (5/3, 7/3)

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := 10 * x - 5 * y = 5

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := 8 * x + 2 * y = 18

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem intersection_point_unique :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l3590_359014


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3590_359092

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 2) % 12 = 0 ∧
  (n - 2) % 16 = 0 ∧
  (n - 2) % 18 = 0 ∧
  (n - 2) % 21 = 0 ∧
  (n - 2) % 28 = 0 ∧
  (n - 2) % 32 = 0 ∧
  (n - 2) % 45 = 0

def is_sum_of_consecutive_primes (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ Nat.Prime (p + 1) ∧ n = p + (p + 1)

theorem smallest_number_satisfying_conditions :
  (is_divisible_by_all 10090 ∧ is_sum_of_consecutive_primes 10090) ∧
  ∀ m : ℕ, m < 10090 → ¬(is_divisible_by_all m ∧ is_sum_of_consecutive_primes m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3590_359092


namespace NUMINAMATH_CALUDE_card_probability_ratio_l3590_359001

def num_cards : ℕ := 40
def num_numbers : ℕ := 10
def cards_per_number : ℕ := 4
def cards_drawn : ℕ := 4

def p : ℚ := num_numbers / (num_cards.choose cards_drawn)
def q : ℚ := (num_numbers * (num_numbers - 1) * (cards_per_number.choose 3) * (cards_per_number.choose 1)) / (num_cards.choose cards_drawn)

theorem card_probability_ratio : q / p = 144 := by
  sorry

end NUMINAMATH_CALUDE_card_probability_ratio_l3590_359001


namespace NUMINAMATH_CALUDE_vlecks_for_45_degrees_l3590_359062

/-- The number of vlecks in a full circle on Venus. -/
def full_circle_vlecks : ℕ := 600

/-- The number of degrees in a full circle on Earth. -/
def full_circle_degrees : ℕ := 360

/-- Converts an angle in degrees to vlecks. -/
def degrees_to_vlecks (degrees : ℚ) : ℚ :=
  (degrees / full_circle_degrees) * full_circle_vlecks

/-- Theorem: 45 degrees corresponds to 75 vlecks on Venus. -/
theorem vlecks_for_45_degrees : degrees_to_vlecks 45 = 75 := by
  sorry

end NUMINAMATH_CALUDE_vlecks_for_45_degrees_l3590_359062


namespace NUMINAMATH_CALUDE_inverse_exponential_point_l3590_359080

theorem inverse_exponential_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (Function.invFun (fun x ↦ a^x) 9 = 2) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_exponential_point_l3590_359080


namespace NUMINAMATH_CALUDE_sqrt_division_equality_l3590_359083

theorem sqrt_division_equality : Real.sqrt 10 / Real.sqrt 5 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_division_equality_l3590_359083


namespace NUMINAMATH_CALUDE_triangle_area_with_base_12_height_15_l3590_359008

/-- The area of a triangle with base 12 and height 15 is 90 -/
theorem triangle_area_with_base_12_height_15 :
  let base : ℝ := 12
  let height : ℝ := 15
  let area : ℝ := (1 / 2) * base * height
  area = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_base_12_height_15_l3590_359008


namespace NUMINAMATH_CALUDE_xyz_equals_ten_l3590_359033

theorem xyz_equals_ten (a b c x y z : ℂ) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h_a : a = (b + c) / (x - 3))
  (h_b : b = (a + c) / (y - 3))
  (h_c : c = (a + b) / (z - 3))
  (h_sum_prod : x * y + x * z + y * z = 7)
  (h_sum : x + y + z = 4) :
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_ten_l3590_359033


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l3590_359022

theorem imaginary_part_of_complex_division : 
  let z : ℂ := 1 / (2 + Complex.I)
  Complex.im z = -1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l3590_359022


namespace NUMINAMATH_CALUDE_calculate_gladys_speed_l3590_359087

def team_size : Nat := 5

def rudy_speed : Nat := 64
def joyce_speed : Nat := 76
def lisa_speed : Nat := 80
def mike_speed : Nat := 89

def team_average : Nat := 80

def gladys_speed : Nat := 91

theorem calculate_gladys_speed :
  team_size * team_average - (rudy_speed + joyce_speed + lisa_speed + mike_speed) = gladys_speed := by
  sorry

end NUMINAMATH_CALUDE_calculate_gladys_speed_l3590_359087


namespace NUMINAMATH_CALUDE_total_albums_l3590_359039

theorem total_albums (adele bridget katrina miriam : ℕ) : 
  adele = 30 →
  bridget = adele - 15 →
  katrina = 6 * bridget →
  miriam = 5 * katrina →
  adele + bridget + katrina + miriam = 585 :=
by
  sorry

end NUMINAMATH_CALUDE_total_albums_l3590_359039


namespace NUMINAMATH_CALUDE_opposite_of_2023_l3590_359099

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (n : ℤ) : ℤ := -n

/-- Theorem: The opposite of 2023 is -2023. -/
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l3590_359099


namespace NUMINAMATH_CALUDE_swamp_ecosystem_flies_eaten_l3590_359069

/-- Represents the number of flies eaten daily in a swamp ecosystem -/
def flies_eaten_daily (num_gharials : ℕ) (fish_per_gharial : ℕ) (frogs_per_fish : ℕ) (flies_per_frog : ℕ) : ℕ :=
  num_gharials * fish_per_gharial * frogs_per_fish * flies_per_frog

/-- Theorem stating the number of flies eaten daily in the given swamp ecosystem -/
theorem swamp_ecosystem_flies_eaten :
  flies_eaten_daily 9 15 8 30 = 32400 := by
  sorry

#eval flies_eaten_daily 9 15 8 30

end NUMINAMATH_CALUDE_swamp_ecosystem_flies_eaten_l3590_359069


namespace NUMINAMATH_CALUDE_cd_equals_three_plus_b_l3590_359035

theorem cd_equals_three_plus_b 
  (a b c d : ℝ) 
  (h1 : a + b = 11) 
  (h2 : b + c = 9) 
  (h3 : a + d = 5) : 
  c + d = 3 + b := by
sorry

end NUMINAMATH_CALUDE_cd_equals_three_plus_b_l3590_359035


namespace NUMINAMATH_CALUDE_quotient_problem_l3590_359029

theorem quotient_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 165)
  (h2 : divisor = 18)
  (h3 : remainder = 3)
  (h4 : dividend = quotient * divisor + remainder) :
  quotient = 9 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l3590_359029


namespace NUMINAMATH_CALUDE_cliffs_rock_collection_l3590_359084

/-- The number of rocks in Cliff's collection -/
def total_rocks (igneous sedimentary metamorphic comet : ℕ) : ℕ :=
  igneous + sedimentary + metamorphic + comet

theorem cliffs_rock_collection :
  ∀ (igneous sedimentary metamorphic comet : ℕ),
    igneous = sedimentary / 2 →
    metamorphic = igneous / 3 →
    comet = 2 * metamorphic →
    igneous / 4 = 15 →
    comet / 2 = 20 →
    total_rocks igneous sedimentary metamorphic comet = 240 := by
  sorry

end NUMINAMATH_CALUDE_cliffs_rock_collection_l3590_359084


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l3590_359067

theorem two_digit_number_sum (n : ℕ) : 
  (10 ≤ n ∧ n < 100) →  -- n is a two-digit number
  (n / 2 : ℚ) = (n / 4 : ℚ) + 3 →  -- one half of n exceeds its one fourth by 3
  (n / 10 + n % 10 = 3) :=  -- sum of digits is 3
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l3590_359067


namespace NUMINAMATH_CALUDE_f_domain_and_range_l3590_359065

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt (1 - Real.cos (2 * x) + 2 * Real.sin x) + 1 / Real.sqrt (Real.sin x ^ 2 + Real.sin x)

def domain (x : ℝ) : Prop := ∃ k : ℤ, 2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi

theorem f_domain_and_range :
  (∀ x : ℝ, f x ≠ 0 → domain x) ∧
  (∀ y : ℝ, y ≥ 2 * (2 : ℝ) ^ (1/4) → ∃ x : ℝ, f x = y) :=
sorry

end NUMINAMATH_CALUDE_f_domain_and_range_l3590_359065


namespace NUMINAMATH_CALUDE_jar_to_pot_ratio_l3590_359066

/-- Proves that the ratio of jars to clay pots is 2:1 given the problem conditions --/
theorem jar_to_pot_ratio :
  ∀ (num_pots : ℕ),
  (∃ (k : ℕ), 16 = k * num_pots) →
  16 * 5 + num_pots * (5 * 3) = 200 →
  (16 : ℚ) / num_pots = 2 := by
  sorry

end NUMINAMATH_CALUDE_jar_to_pot_ratio_l3590_359066


namespace NUMINAMATH_CALUDE_range_of_4a_minus_2b_l3590_359054

theorem range_of_4a_minus_2b (a b : ℝ) 
  (h1 : 1 ≤ a - b) (h2 : a - b ≤ 2) 
  (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) : 
  5 ≤ 4*a - 2*b ∧ 4*a - 2*b ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_range_of_4a_minus_2b_l3590_359054


namespace NUMINAMATH_CALUDE_range_of_t_for_perpendicular_lines_l3590_359018

/-- Circle C with center (1,4) and radius √10 -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 4)^2 = 10

/-- Point M with coordinates (5,t) -/
def point_M (t : ℝ) : ℝ × ℝ := (5, t)

/-- Theorem stating the range of t for which there exist points A and B on circle C
    such that MA ⊥ MB -/
theorem range_of_t_for_perpendicular_lines :
  ∀ t : ℝ, (∃ A B : ℝ × ℝ, 
    circle_C A.1 A.2 ∧ 
    circle_C B.1 B.2 ∧ 
    (A.1 - 5) * (B.1 - 5) + (A.2 - t) * (B.2 - t) = 0) 
  ↔ 2 ≤ t ∧ t ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_t_for_perpendicular_lines_l3590_359018


namespace NUMINAMATH_CALUDE_sin_lt_tan_in_first_quadrant_half_angle_in_first_or_third_quadrant_sin_not_always_four_fifths_sector_angle_is_one_radian_l3590_359015

-- Statement ①
theorem sin_lt_tan_in_first_quadrant (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  Real.sin α < Real.tan α := by sorry

-- Statement ②
theorem half_angle_in_first_or_third_quadrant (α : Real) 
  (h : Real.pi / 2 < α ∧ α < Real.pi) :
  (0 < α / 2 ∧ α / 2 < Real.pi / 2) ∨ 
  (Real.pi < α / 2 ∧ α / 2 < 3 * Real.pi / 2) := by sorry

-- Statement ③ (incorrect)
theorem sin_not_always_four_fifths (k : Real) (h : k ≠ 0) :
  ∃ α, Real.cos α = 3 * k / 5 ∧ Real.sin α = 4 * k / 5 ∧ Real.sin α ≠ 4 / 5 := by sorry

-- Statement ④
theorem sector_angle_is_one_radian (perimeter radius : Real) 
  (h1 : perimeter = 6) (h2 : radius = 2) :
  (perimeter - 2 * radius) / radius = 1 := by sorry

end NUMINAMATH_CALUDE_sin_lt_tan_in_first_quadrant_half_angle_in_first_or_third_quadrant_sin_not_always_four_fifths_sector_angle_is_one_radian_l3590_359015


namespace NUMINAMATH_CALUDE_max_projection_area_is_one_l3590_359074

/-- A tetrahedron with specific properties -/
structure SpecialTetrahedron where
  /-- Two adjacent faces are isosceles right triangles -/
  isosceles_right_faces : Bool
  /-- The hypotenuse of the isosceles right triangles is 2 -/
  hypotenuse : ℝ
  /-- The dihedral angle between the two adjacent faces is 60 degrees -/
  dihedral_angle : ℝ
  /-- The tetrahedron rotates around the common edge of the two faces -/
  rotates_around_common_edge : Bool

/-- The maximum projection area of the rotating tetrahedron -/
def max_projection_area (t : SpecialTetrahedron) : ℝ := sorry

/-- Theorem stating that the maximum projection area is 1 -/
theorem max_projection_area_is_one (t : SpecialTetrahedron) 
  (h1 : t.isosceles_right_faces = true)
  (h2 : t.hypotenuse = 2)
  (h3 : t.dihedral_angle = Real.pi / 3)  -- 60 degrees in radians
  (h4 : t.rotates_around_common_edge = true) :
  max_projection_area t = 1 := by sorry

end NUMINAMATH_CALUDE_max_projection_area_is_one_l3590_359074


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3590_359072

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (∀ θ : ℝ, θ = 150 ∧ θ = (n - 2 : ℝ) * 180 / n) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3590_359072


namespace NUMINAMATH_CALUDE_tileD_in_rectangleII_l3590_359053

-- Define the structure for a tile
structure Tile where
  top : Nat
  right : Nat
  bottom : Nat
  left : Nat

-- Define the tiles
def tileA : Tile := ⟨3, 5, 2, 0⟩
def tileB : Tile := ⟨2, 0, 5, 3⟩
def tileC : Tile := ⟨5, 3, 1, 2⟩
def tileD : Tile := ⟨0, 1, 3, 5⟩

-- Define a function to check if two tiles match on their adjacent sides
def matchTiles (t1 t2 : Tile) (side : Nat) : Prop :=
  match side with
  | 0 => t1.right = t2.left   -- Right of t1 matches Left of t2
  | 1 => t1.bottom = t2.top   -- Bottom of t1 matches Top of t2
  | 2 => t1.left = t2.right   -- Left of t1 matches Right of t2
  | 3 => t1.top = t2.bottom   -- Top of t1 matches Bottom of t2
  | _ => False

-- Theorem stating that Tile D must be in Rectangle II
theorem tileD_in_rectangleII : ∃ (t1 t2 t3 : Tile), 
  (t1 = tileA ∨ t1 = tileB ∨ t1 = tileC) ∧
  (t2 = tileA ∨ t2 = tileB ∨ t2 = tileC) ∧
  (t3 = tileA ∨ t3 = tileB ∨ t3 = tileC) ∧
  t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧
  matchTiles t1 tileD 0 ∧
  matchTiles tileD t2 0 ∧
  matchTiles t3 tileD 3 :=
by sorry

end NUMINAMATH_CALUDE_tileD_in_rectangleII_l3590_359053


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l3590_359050

theorem perfect_square_trinomial_m_values (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 2*(m+1)*x + 25 = (x + a)^2) → 
  m = 4 ∨ m = -6 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l3590_359050


namespace NUMINAMATH_CALUDE_chord_length_dot_product_value_l3590_359086

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x + y - 6 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 0)^2 + (y - 1)^2 = 5

-- Define point P
def point_P : ℝ × ℝ := (0, -2)

-- Theorem for the length of the chord
theorem chord_length :
  ∃ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 10 :=
sorry

-- Theorem for the dot product
theorem dot_product_value :
  ∀ (A B : ℝ × ℝ),
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    A ≠ B ∧
    ∃ (t : ℝ), A.1 = point_P.1 + t * (A.1 - point_P.1) ∧
                A.2 = point_P.2 + t * (A.2 - point_P.2) ∧
                B.1 = point_P.1 + t * (B.1 - point_P.1) ∧
                B.2 = point_P.2 + t * (B.2 - point_P.2) →
    ((A.1 - point_P.1) * (B.1 - point_P.1) + (A.2 - point_P.2) * (B.2 - point_P.2))^2 = 16 :=
sorry

end NUMINAMATH_CALUDE_chord_length_dot_product_value_l3590_359086


namespace NUMINAMATH_CALUDE_train_speed_l3590_359019

/-- The speed of a train given its length, the speed of a man running in the opposite direction,
    and the time it takes for the train to pass the man. -/
theorem train_speed (train_length : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_length = 140 →
  man_speed = 6 →
  passing_time = 6 →
  (train_length / passing_time) * 3.6 - man_speed = 78 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3590_359019


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3590_359026

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.sin y = 2023)
  (eq2 : x + 2023 * Real.cos y = 2021)
  (y_range : π/4 ≤ y ∧ y ≤ 3*π/4) :
  x + y = 2023 - Real.sqrt 2 / 2 + 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3590_359026


namespace NUMINAMATH_CALUDE_subtraction_example_l3590_359077

theorem subtraction_example : (3.75 : ℝ) - 1.46 = 2.29 := by sorry

end NUMINAMATH_CALUDE_subtraction_example_l3590_359077


namespace NUMINAMATH_CALUDE_total_students_is_1076_l3590_359005

/-- Represents the number of students in a school --/
structure School where
  girls : ℕ
  boys : ℕ

/-- The total number of students in the school --/
def School.total (s : School) : ℕ := s.girls + s.boys

/-- A school with 402 more girls than boys and 739 girls --/
def our_school : School := {
  girls := 739,
  boys := 739 - 402
}

/-- Theorem stating that the total number of students in our_school is 1076 --/
theorem total_students_is_1076 : our_school.total = 1076 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_1076_l3590_359005


namespace NUMINAMATH_CALUDE_complex_modulus_equal_parts_l3590_359003

theorem complex_modulus_equal_parts (a : ℝ) :
  let z : ℂ := (1 + 2*I) * (a + I)
  (z.re = z.im) → Complex.abs z = 5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_equal_parts_l3590_359003


namespace NUMINAMATH_CALUDE_not_exp_ix_always_one_l3590_359078

open Complex

theorem not_exp_ix_always_one (x : ℝ) : ¬ ∀ x, exp (I * x) = 1 := by
  sorry

/-- e^(ix) is a periodic function with period 2π -/
axiom exp_ix_periodic : ∀ x : ℝ, exp (I * x) = exp (I * (x + 2 * Real.pi))

/-- e^(ix) = e^(i(x + 2πk)) for any integer k -/
axiom exp_ix_shift : ∀ (x : ℝ) (k : ℤ), exp (I * x) = exp (I * (x + 2 * Real.pi * ↑k))

end NUMINAMATH_CALUDE_not_exp_ix_always_one_l3590_359078


namespace NUMINAMATH_CALUDE_sum_remainder_modulo_9_l3590_359073

theorem sum_remainder_modulo_9 : 
  (8 + 77 + 666 + 5555 + 44444 + 333333 + 2222222 + 11111111) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_modulo_9_l3590_359073


namespace NUMINAMATH_CALUDE_sin_780_degrees_l3590_359094

theorem sin_780_degrees : Real.sin (780 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_780_degrees_l3590_359094


namespace NUMINAMATH_CALUDE_initial_bird_families_l3590_359000

theorem initial_bird_families (flew_away left_now : ℕ) 
  (h1 : flew_away = 27) 
  (h2 : left_now = 14) : 
  flew_away + left_now = 41 := by
  sorry

end NUMINAMATH_CALUDE_initial_bird_families_l3590_359000


namespace NUMINAMATH_CALUDE_inequality_proof_l3590_359095

theorem inequality_proof (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  (a * b^n)^(1/(n+1 : ℝ)) < (a + n * b) / (n + 1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3590_359095


namespace NUMINAMATH_CALUDE_photo_arrangements_l3590_359079

def num_boys : ℕ := 4
def num_girls : ℕ := 3

def arrangements_girls_at_ends : ℕ := 720
def arrangements_no_adjacent_girls : ℕ := 1440
def arrangements_girl_A_right_of_B : ℕ := 2520

theorem photo_arrangements :
  (num_boys = 4 ∧ num_girls = 3) →
  (arrangements_girls_at_ends = 720 ∧
   arrangements_no_adjacent_girls = 1440 ∧
   arrangements_girl_A_right_of_B = 2520) :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l3590_359079


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3590_359041

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 16*y = x*y) :
  ∀ z w : ℝ, z > 0 → w > 0 → z + 16*w = z*w → x + y ≤ z + w ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 16*b = a*b ∧ a + b = 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3590_359041


namespace NUMINAMATH_CALUDE_fraction_simplification_l3590_359090

theorem fraction_simplification : 
  (1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3590_359090


namespace NUMINAMATH_CALUDE_sandys_phone_bill_l3590_359071

theorem sandys_phone_bill (kim_age : ℕ) (sandy_age : ℕ) (sandy_bill : ℕ) : 
  kim_age = 10 →
  sandy_age + 2 = 3 * (kim_age + 2) →
  sandy_bill = 10 * sandy_age →
  sandy_bill = 340 :=
by
  sorry

end NUMINAMATH_CALUDE_sandys_phone_bill_l3590_359071


namespace NUMINAMATH_CALUDE_f_properties_l3590_359049

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom func_property : ∀ x y : ℝ, f x * f y = f (x + y - 1)
axiom greater_than_one : ∀ x : ℝ, x > 1 → f x > 1

-- State the theorem
theorem f_properties : 
  (f 1 = 1) ∧ 
  (∀ x : ℝ, x < 1 → 0 < f x ∧ f x < 1) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3590_359049


namespace NUMINAMATH_CALUDE_probability_real_roots_l3590_359058

/-- The probability that the equation x^2 - mx + 4 = 0 has real roots,
    given that m is uniformly distributed in the interval [0, 6]. -/
theorem probability_real_roots : ℝ := by
  sorry

#check probability_real_roots

end NUMINAMATH_CALUDE_probability_real_roots_l3590_359058


namespace NUMINAMATH_CALUDE_pizza_combinations_l3590_359055

/-- The number of available toppings -/
def num_toppings : ℕ := 8

/-- The number of toppings on a pizza of type 1 -/
def toppings_type1 : ℕ := 4

/-- The number of toppings on a pizza of type 2 -/
def toppings_type2 : ℕ := 3

/-- Theorem: The sum of 4-topping and 3-topping pizza combinations with 8 available toppings is 126 -/
theorem pizza_combinations : 
  (Nat.choose num_toppings toppings_type1) + (Nat.choose num_toppings toppings_type2) = 126 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l3590_359055


namespace NUMINAMATH_CALUDE_parabola_equation_l3590_359011

/-- A parabola is defined by its vertex and a point it passes through. -/
structure Parabola where
  vertex : ℝ × ℝ
  point : ℝ × ℝ

/-- The analytical expression of a parabola. -/
def parabola_expression (p : Parabola) : ℝ → ℝ :=
  fun x => -(x + 2)^2 + 3

theorem parabola_equation (p : Parabola) 
  (h1 : p.vertex = (-2, 3)) 
  (h2 : p.point = (1, -6)) : 
  ∀ x, parabola_expression p x = -(x + 2)^2 + 3 := by
  sorry

#check parabola_equation

end NUMINAMATH_CALUDE_parabola_equation_l3590_359011


namespace NUMINAMATH_CALUDE_car_selection_problem_l3590_359093

theorem car_selection_problem (cars : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) 
  (h1 : cars = 12)
  (h2 : selections_per_client = 4)
  (h3 : selections_per_car = 3) :
  (cars * selections_per_car) / selections_per_client = 9 := by
  sorry

end NUMINAMATH_CALUDE_car_selection_problem_l3590_359093


namespace NUMINAMATH_CALUDE_infinitely_many_perfect_squares_implies_divisibility_l3590_359064

theorem infinitely_many_perfect_squares_implies_divisibility 
  (a b : ℕ+) 
  (h : ∃ (S : Set (ℕ+ × ℕ+)), Set.Infinite S ∧ 
    ∀ (p : ℕ+ × ℕ+), p ∈ S → 
      ∃ (r s : ℕ+), (p.1.val^2 + a.val * p.2.val + b.val = r.val^2) ∧ 
                     (p.2.val^2 + a.val * p.1.val + b.val = s.val^2)) : 
  a.val ∣ (2 * b.val) := by
sorry

end NUMINAMATH_CALUDE_infinitely_many_perfect_squares_implies_divisibility_l3590_359064


namespace NUMINAMATH_CALUDE_paco_initial_sweet_cookies_l3590_359063

/-- The number of sweet cookies Paco had initially -/
def initial_sweet_cookies : ℕ := sorry

/-- The number of sweet cookies Paco ate -/
def eaten_sweet_cookies : ℕ := 15

/-- The number of sweet cookies Paco had left -/
def remaining_sweet_cookies : ℕ := 7

/-- Theorem: Paco had 22 sweet cookies initially -/
theorem paco_initial_sweet_cookies :
  initial_sweet_cookies = eaten_sweet_cookies + remaining_sweet_cookies ∧
  initial_sweet_cookies = 22 :=
by sorry

end NUMINAMATH_CALUDE_paco_initial_sweet_cookies_l3590_359063


namespace NUMINAMATH_CALUDE_prob_at_least_two_same_correct_l3590_359032

/-- The number of sides on each die -/
def num_sides : Nat := 8

/-- The number of dice rolled -/
def num_dice : Nat := 7

/-- The probability of rolling 7 fair 8-sided dice and getting at least two dice showing the same number -/
def prob_at_least_two_same : ℚ := 319 / 320

/-- Theorem stating that the probability of at least two dice showing the same number
    when rolling 7 fair 8-sided dice is equal to 319/320 -/
theorem prob_at_least_two_same_correct :
  (1 : ℚ) - (Nat.factorial num_sides / Nat.factorial (num_sides - num_dice)) / (num_sides ^ num_dice) = prob_at_least_two_same := by
  sorry


end NUMINAMATH_CALUDE_prob_at_least_two_same_correct_l3590_359032


namespace NUMINAMATH_CALUDE_parallel_vectors_cos_2alpha_l3590_359085

theorem parallel_vectors_cos_2alpha (α : ℝ) :
  let a : ℝ × ℝ := (1/3, Real.tan α)
  let b : ℝ × ℝ := (Real.cos α, 1)
  (∃ (k : ℝ), a = k • b) → Real.cos (2 * α) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_cos_2alpha_l3590_359085


namespace NUMINAMATH_CALUDE_z_plus_two_over_z_traces_ellipse_l3590_359060

/-- Given a complex number z with |z| = 3, prove that z + 2/z traces an ellipse -/
theorem z_plus_two_over_z_traces_ellipse (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  ∀ (w : ℂ), w = z + 2 / z → (w.re / a)^2 + (w.im / b)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_z_plus_two_over_z_traces_ellipse_l3590_359060


namespace NUMINAMATH_CALUDE_prob_same_color_specific_l3590_359002

/-- Probability of drawing two marbles of the same color -/
def prob_same_color (red white blue green : ℕ) : ℚ :=
  let total := red + white + blue + green
  let prob_red := (red * (red - 1)) / (total * (total - 1))
  let prob_white := (white * (white - 1)) / (total * (total - 1))
  let prob_blue := (blue * (blue - 1)) / (total * (total - 1))
  let prob_green := (green * (green - 1)) / (total * (total - 1))
  prob_red + prob_white + prob_blue + prob_green

theorem prob_same_color_specific : prob_same_color 5 6 7 3 = 7 / 30 := by
  sorry

#eval prob_same_color 5 6 7 3

end NUMINAMATH_CALUDE_prob_same_color_specific_l3590_359002


namespace NUMINAMATH_CALUDE_cubic_inequality_l3590_359068

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 + 27*x > 0 ↔ x ∈ Set.union (Set.Ioo 0 3) (Set.Ioi 9) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3590_359068


namespace NUMINAMATH_CALUDE_flower_shop_optimal_strategy_l3590_359020

/-- Represents the flower shop's sales and profit model -/
structure FlowerShop where
  cost : ℝ := 50
  max_margin : ℝ := 0.52
  sales : ℝ → ℝ
  profit : ℝ → ℝ
  profit_after_donation : ℝ → ℝ → ℝ

/-- The main theorem about the flower shop's optimal pricing and donation strategy -/
theorem flower_shop_optimal_strategy (shop : FlowerShop) 
  (h_sales : ∀ x, shop.sales x = -6 * x + 600) 
  (h_profit : ∀ x, shop.profit x = (x - shop.cost) * shop.sales x) 
  (h_profit_donation : ∀ x n, shop.profit_after_donation x n = shop.profit x - n * shop.sales x) 
  (h_price_range : ∀ x, x ≥ shop.cost ∧ x ≤ shop.cost * (1 + shop.max_margin)) :
  (∃ max_profit : ℝ, max_profit = 3750 ∧ 
    ∀ x, shop.profit x ≤ max_profit ∧ 
    (shop.profit 75 = max_profit)) ∧
  (∀ n, (∀ x₁ x₂, x₁ < x₂ → shop.profit_after_donation x₁ n < shop.profit_after_donation x₂ n) 
    ↔ (1 < n ∧ n < 2)) :=
sorry

end NUMINAMATH_CALUDE_flower_shop_optimal_strategy_l3590_359020


namespace NUMINAMATH_CALUDE_ten_steps_climb_l3590_359042

/-- Number of ways to climb n steps when allowed to take 1, 2, or 3 steps at a time -/
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | k + 3 => climbStairs (k + 2) + climbStairs (k + 1) + climbStairs k

/-- Theorem stating that there are 274 ways to climb 10 steps -/
theorem ten_steps_climb : climbStairs 10 = 274 := by
  sorry


end NUMINAMATH_CALUDE_ten_steps_climb_l3590_359042


namespace NUMINAMATH_CALUDE_factor_probability_l3590_359046

/-- The number of consecutive natural numbers in the set -/
def n : ℕ := 120

/-- The factorial we're considering -/
def f : ℕ := 5

/-- The number of factors of f! -/
def num_factors : ℕ := 16

/-- The probability of selecting a factor of f! from the set of n consecutive natural numbers -/
def probability : ℚ := num_factors / n

theorem factor_probability : probability = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_factor_probability_l3590_359046


namespace NUMINAMATH_CALUDE_difference_of_squares_application_l3590_359016

theorem difference_of_squares_application (a b : ℝ) :
  (1/4 * a + b) * (b - 1/4 * a) = b^2 - (1/16) * a^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_application_l3590_359016


namespace NUMINAMATH_CALUDE_exponent_simplification_l3590_359057

theorem exponent_simplification (a : ℝ) (h : a > 0) : 
  a^(1/2) * a^(2/3) / a^(1/6) = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l3590_359057


namespace NUMINAMATH_CALUDE_expression_simplification_l3590_359027

theorem expression_simplification (x : ℝ) : 
  3*x - 3*(2 - x) + 4*(2 + 3*x) - 5*(1 - 2*x) = 28*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3590_359027


namespace NUMINAMATH_CALUDE_midpoint_sum_after_doubling_x_l3590_359089

/-- Given a segment with endpoints (10, 3) and (-4, 7), prove that the sum of the doubled x-coordinate
and the y-coordinate of the midpoint is 11. -/
theorem midpoint_sum_after_doubling_x : 
  let p1 : ℝ × ℝ := (10, 3)
  let p2 : ℝ × ℝ := (-4, 7)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let doubled_x : ℝ := 2 * midpoint.1
  doubled_x + midpoint.2 = 11 := by sorry

end NUMINAMATH_CALUDE_midpoint_sum_after_doubling_x_l3590_359089


namespace NUMINAMATH_CALUDE_matrix_determinant_l3590_359017

theorem matrix_determinant : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![9/2, 4; -3/2, 5/2]
  Matrix.det A = 69/4 := by
sorry

end NUMINAMATH_CALUDE_matrix_determinant_l3590_359017


namespace NUMINAMATH_CALUDE_four_digit_difference_l3590_359098

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ n / 1000 = 7

def reverse_last_three_digits (n : ℕ) : ℕ :=
  let a := (n / 100) % 10
  let b := (n / 10) % 10
  let c := n % 10
  1000 * c + 100 * b + 10 * a + 7

theorem four_digit_difference (n : ℕ) : 
  is_valid_number n → n = reverse_last_three_digits n + 3546 → 
  n = 7053 ∨ n = 7163 ∨ n = 7273 ∨ n = 7383 ∨ n = 7493 :=
sorry

end NUMINAMATH_CALUDE_four_digit_difference_l3590_359098


namespace NUMINAMATH_CALUDE_whale_population_prediction_l3590_359034

theorem whale_population_prediction (whales_last_year whales_this_year whales_next_year predicted_increase : ℕ) : 
  whales_last_year = 4000 →
  whales_this_year = 2 * whales_last_year →
  predicted_increase = 800 →
  whales_next_year = whales_this_year + predicted_increase →
  whales_next_year = 8800 := by
sorry

end NUMINAMATH_CALUDE_whale_population_prediction_l3590_359034


namespace NUMINAMATH_CALUDE_trig_sum_equals_one_l3590_359037

theorem trig_sum_equals_one : 
  Real.sin (300 * Real.pi / 180) + Real.cos (390 * Real.pi / 180) + Real.tan (-135 * Real.pi / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equals_one_l3590_359037


namespace NUMINAMATH_CALUDE_friend_savings_rate_l3590_359075

/-- Proves that given the initial amounts and saving rates, the friend's weekly savings
    that result in equal total savings after 25 weeks is 5 dollars. -/
theorem friend_savings_rate (your_initial : ℕ) (your_weekly : ℕ) (friend_initial : ℕ) (weeks : ℕ) :
  your_initial = 160 →
  your_weekly = 7 →
  friend_initial = 210 →
  weeks = 25 →
  ∃ (friend_weekly : ℕ),
    your_initial + your_weekly * weeks = friend_initial + friend_weekly * weeks ∧
    friend_weekly = 5 :=
by sorry

end NUMINAMATH_CALUDE_friend_savings_rate_l3590_359075


namespace NUMINAMATH_CALUDE_choose_starters_count_l3590_359048

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 7 starters from a team of 16 players,
    where exactly one player must be chosen from a set of 4 quadruplets -/
def choose_starters : ℕ :=
  4 * binomial 12 6

theorem choose_starters_count : choose_starters = 3696 := by sorry

end NUMINAMATH_CALUDE_choose_starters_count_l3590_359048


namespace NUMINAMATH_CALUDE_min_value_expression_l3590_359047

theorem min_value_expression (x y : ℝ) : 3 * x^2 + 3 * x * y + y^2 - 6 * x + 4 * y + 5 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3590_359047


namespace NUMINAMATH_CALUDE_alternating_squares_sum_l3590_359038

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 268 := by
  sorry

end NUMINAMATH_CALUDE_alternating_squares_sum_l3590_359038


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3590_359007

-- Define the sides of the triangle
def side1 : ℝ := 9
def side2 : ℝ := 9
def side3 : ℝ := 4

-- Define the isosceles triangle condition
def is_isosceles (a b c : ℝ) : Prop := (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

-- Define the triangle inequality
def satisfies_triangle_inequality (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  is_isosceles side1 side2 side3 ∧
  satisfies_triangle_inequality side1 side2 side3 →
  perimeter side1 side2 side3 = 22 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3590_359007


namespace NUMINAMATH_CALUDE_probability_at_least_four_out_of_five_l3590_359025

theorem probability_at_least_four_out_of_five (p : ℝ) (h : p = 4/5) :
  let binomial (n k : ℕ) := Nat.choose n k
  let prob_exactly (k : ℕ) := (binomial 5 k : ℝ) * p^k * (1 - p)^(5 - k)
  prob_exactly 4 + prob_exactly 5 = 2304/3125 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_four_out_of_five_l3590_359025


namespace NUMINAMATH_CALUDE_square_perimeter_l3590_359044

/-- The perimeter of a square is equal to four times its side length. -/
theorem square_perimeter (side : ℝ) (h : side = 13) : 4 * side = 52 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3590_359044


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3590_359030

theorem quadratic_form_sum (x : ℝ) : 
  ∃ (a b c : ℝ), (6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) ∧ (a + b + c = 171) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3590_359030


namespace NUMINAMATH_CALUDE_ages_cube_sum_l3590_359082

theorem ages_cube_sum (r j m : ℕ) : 
  (5 * r + 2 * j = 3 * m) →
  (3 * m^2 + 2 * j^2 = 5 * r^2) →
  (Nat.gcd r j = 1 ∧ Nat.gcd j m = 1 ∧ Nat.gcd r m = 1) →
  r^3 + j^3 + m^3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_ages_cube_sum_l3590_359082


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3590_359012

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, x > 1 → x^2 + 2*x > 0) ∧
  (∃ x : ℝ, x^2 + 2*x > 0 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3590_359012


namespace NUMINAMATH_CALUDE_wedding_decoration_cost_per_place_setting_l3590_359004

/-- Calculates the cost per place setting for wedding decorations --/
theorem wedding_decoration_cost_per_place_setting 
  (num_tables : ℕ) 
  (tablecloth_cost : ℕ) 
  (place_settings_per_table : ℕ) 
  (roses_per_centerpiece : ℕ) 
  (rose_cost : ℕ) 
  (lilies_per_centerpiece : ℕ) 
  (lily_cost : ℕ) 
  (total_decoration_cost : ℕ) : 
  num_tables = 20 →
  tablecloth_cost = 25 →
  place_settings_per_table = 4 →
  roses_per_centerpiece = 10 →
  rose_cost = 5 →
  lilies_per_centerpiece = 15 →
  lily_cost = 4 →
  total_decoration_cost = 3500 →
  (total_decoration_cost - 
   (num_tables * tablecloth_cost + 
    num_tables * (roses_per_centerpiece * rose_cost + lilies_per_centerpiece * lily_cost))) / 
   (num_tables * place_settings_per_table) = 10 := by
  sorry

end NUMINAMATH_CALUDE_wedding_decoration_cost_per_place_setting_l3590_359004


namespace NUMINAMATH_CALUDE_percent_relationship_l3590_359070

theorem percent_relationship (x y z : ℝ) (h1 : x = 1.20 * y) (h2 : y = 0.70 * z) :
  x = 0.84 * z := by sorry

end NUMINAMATH_CALUDE_percent_relationship_l3590_359070


namespace NUMINAMATH_CALUDE_beef_weight_loss_percentage_l3590_359056

/-- Proves that a side of beef with given initial and final weights loses approximately 30% of its weight during processing -/
theorem beef_weight_loss_percentage (initial_weight final_weight : ℝ) 
  (h1 : initial_weight = 714.2857142857143)
  (h2 : final_weight = 500) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |((initial_weight - final_weight) / initial_weight * 100) - 30| < ε :=
sorry

end NUMINAMATH_CALUDE_beef_weight_loss_percentage_l3590_359056


namespace NUMINAMATH_CALUDE_mass_percentage_K_is_23_81_l3590_359088

/-- The mass percentage of K in a compound -/
def mass_percentage_K : ℝ := 23.81

/-- Theorem stating that the mass percentage of K in the compound is 23.81% -/
theorem mass_percentage_K_is_23_81 :
  mass_percentage_K = 23.81 := by sorry

end NUMINAMATH_CALUDE_mass_percentage_K_is_23_81_l3590_359088


namespace NUMINAMATH_CALUDE_g_4_equals_7_5_l3590_359052

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def g (x : ℝ) : ℝ := 1 / (f⁻¹ x) + 7

theorem g_4_equals_7_5 : g 4 = 7.5 := by sorry

end NUMINAMATH_CALUDE_g_4_equals_7_5_l3590_359052


namespace NUMINAMATH_CALUDE_cube_sum_of_symmetric_relations_l3590_359024

theorem cube_sum_of_symmetric_relations (a b c : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a * b + a * c + b * c = 2)
  (h3 : a * b * c = 1) :
  a^3 + b^3 + c^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_of_symmetric_relations_l3590_359024


namespace NUMINAMATH_CALUDE_counterexample_exists_l3590_359009

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem counterexample_exists : ∃ n : ℕ, 
  (sum_of_digits n % 27 = 0) ∧ 
  (n % 27 ≠ 0) ∧ 
  (n = 81 ∨ n = 999 ∨ n = 9918 ∨ n = 18) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3590_359009


namespace NUMINAMATH_CALUDE_second_number_is_72_l3590_359013

theorem second_number_is_72 (a b c : ℚ) : 
  a + b + c = 264 ∧ 
  a = 2 * b ∧ 
  c = (1/3) * a → 
  b = 72 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_72_l3590_359013


namespace NUMINAMATH_CALUDE_insert_two_digits_into_five_digit_number_l3590_359036

/-- The number of ways to insert two indistinguishable digits into a 5-digit number to form a 7-digit number -/
def insert_two_digits (n : ℕ) : ℕ :=
  let total_positions := n + 1
  let total_arrangements := total_positions * total_positions
  let arrangements_together := total_positions
  total_arrangements - arrangements_together

/-- The theorem stating that inserting two indistinguishable digits into a 5-digit number results in 30 different 7-digit numbers -/
theorem insert_two_digits_into_five_digit_number :
  insert_two_digits 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_insert_two_digits_into_five_digit_number_l3590_359036


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomials_l3590_359043

def poly1 (x : ℤ) : ℤ := 2 * x^3 - 3 * x^2 - 11 * x + 6
def poly2 (x : ℤ) : ℤ := x^4 + 4 * x^3 - 9 * x^2 - 16 * x + 20

theorem integer_roots_of_polynomials :
  (∀ x : ℤ, poly1 x = 0 ↔ x = -2 ∨ x = 3) ∧
  (∀ x : ℤ, poly2 x = 0 ↔ x = 1 ∨ x = 2 ∨ x = -2 ∨ x = -5) :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomials_l3590_359043


namespace NUMINAMATH_CALUDE_marks_books_l3590_359021

/-- Given Mark's initial amount, cost per book, and remaining amount, prove the number of books he bought. -/
theorem marks_books (initial_amount : ℕ) (cost_per_book : ℕ) (remaining_amount : ℕ) :
  initial_amount = 85 →
  cost_per_book = 5 →
  remaining_amount = 35 →
  (initial_amount - remaining_amount) / cost_per_book = 10 :=
by sorry

end NUMINAMATH_CALUDE_marks_books_l3590_359021


namespace NUMINAMATH_CALUDE_sin_45_degrees_l3590_359081

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l3590_359081


namespace NUMINAMATH_CALUDE_anika_age_l3590_359091

theorem anika_age :
  ∀ (anika_age maddie_age : ℕ),
  anika_age = (4 * maddie_age) / 3 →
  (anika_age + 15 + maddie_age + 15) / 2 = 50 →
  anika_age = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_anika_age_l3590_359091


namespace NUMINAMATH_CALUDE_filter_kit_cost_difference_l3590_359059

/-- Proves that buying the camera lens filter kit costs more than buying filters individually -/
theorem filter_kit_cost_difference : 
  let kit_price : ℚ := 87.5
  let filter_price_1 : ℚ := 16.45
  let filter_price_2 : ℚ := 14.05
  let filter_price_3 : ℚ := 19.5
  let individual_total : ℚ := 2 * filter_price_1 + 2 * filter_price_2 + filter_price_3
  kit_price - individual_total = 7 :=
by sorry

end NUMINAMATH_CALUDE_filter_kit_cost_difference_l3590_359059


namespace NUMINAMATH_CALUDE_phone_watch_sales_l3590_359023

/-- Represents the total sales amount for two months of phone watch sales -/
def total_sales (x : ℕ) : ℝ := 600 * 60 + 500 * (x - 60)

/-- States that the total sales amount is no less than $86000 -/
def sales_condition (x : ℕ) : Prop := total_sales x ≥ 86000

theorem phone_watch_sales (x : ℕ) : 
  sales_condition x ↔ 600 * 60 + 500 * (x - 60) ≥ 86000 := by sorry

end NUMINAMATH_CALUDE_phone_watch_sales_l3590_359023


namespace NUMINAMATH_CALUDE_oil_weight_in_salad_dressing_salad_dressing_oil_weight_l3590_359045

/-- Calculates the weight of oil per ml in a salad dressing mixture --/
theorem oil_weight_in_salad_dressing 
  (bowl_capacity : ℝ) 
  (oil_proportion : ℝ) 
  (vinegar_proportion : ℝ) 
  (vinegar_weight : ℝ) 
  (total_weight : ℝ) : ℝ :=
  let oil_volume := bowl_capacity * oil_proportion
  let vinegar_volume := bowl_capacity * vinegar_proportion
  let vinegar_total_weight := vinegar_volume * vinegar_weight
  let oil_total_weight := total_weight - vinegar_total_weight
  oil_total_weight / oil_volume

/-- Proves that the weight of oil in the given salad dressing mixture is 5 g/ml --/
theorem salad_dressing_oil_weight :
  oil_weight_in_salad_dressing 150 (2/3) (1/3) 4 700 = 5 := by
  sorry

end NUMINAMATH_CALUDE_oil_weight_in_salad_dressing_salad_dressing_oil_weight_l3590_359045


namespace NUMINAMATH_CALUDE_smallest_candy_count_l3590_359061

theorem smallest_candy_count : ∃ (n : ℕ), 
  100 ≤ n ∧ n < 1000 ∧ 
  (n + 7) % 9 = 0 ∧ 
  (n - 9) % 6 = 0 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < n ∧ (m + 7) % 9 = 0 ∧ (m - 9) % 6 = 0 → False) ∧
  n = 101 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l3590_359061


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3590_359028

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 12/7
  let a₃ : ℚ := 36/7
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → (a₁ * r^(n-1) : ℚ) = 4/7 * 3^(n-1)) →
  r = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3590_359028


namespace NUMINAMATH_CALUDE_valleyball_league_members_l3590_359051

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tshirt_additional_cost : ℕ := 5

/-- The total cost for all members in dollars -/
def total_cost : ℕ := 3300

/-- The number of pairs of socks each member needs -/
def socks_per_member : ℕ := 2

/-- The number of T-shirts each member needs -/
def tshirts_per_member : ℕ := 2

/-- The number of members in the Valleyball Soccer League -/
def number_of_members : ℕ := 97

theorem valleyball_league_members :
  let tshirt_cost := sock_cost + tshirt_additional_cost
  let member_cost := socks_per_member * sock_cost + tshirts_per_member * tshirt_cost
  number_of_members * member_cost = total_cost := by
  sorry


end NUMINAMATH_CALUDE_valleyball_league_members_l3590_359051


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3590_359006

theorem quadratic_root_property (a b s t : ℝ) (h_neq : s ≠ t) 
  (h_ps : s^2 + a*s + b = t) (h_pt : t^2 + a*t + b = s) : 
  (b - s*t)^2 + a*(b - s*t) + b - s*t = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3590_359006


namespace NUMINAMATH_CALUDE_work_rate_proof_l3590_359097

/-- The work rate of person A per day -/
def work_rate_A : ℚ := 1 / 4

/-- The work rate of person B per day -/
def work_rate_B : ℚ := 1 / 2

/-- The work rate of person C per day -/
def work_rate_C : ℚ := 1 / 8

/-- The combined work rate of A, B, and C per day -/
def combined_work_rate : ℚ := work_rate_A + work_rate_B + work_rate_C

theorem work_rate_proof : combined_work_rate = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_work_rate_proof_l3590_359097


namespace NUMINAMATH_CALUDE_number_problem_l3590_359040

theorem number_problem (x y a : ℝ) :
  x * y = 1 →
  (a^((x + y)^2)) / (a^((x - y)^2)) = 1296 →
  a = 6 := by sorry

end NUMINAMATH_CALUDE_number_problem_l3590_359040


namespace NUMINAMATH_CALUDE_smallest_k_for_64k_gt_4_20_l3590_359010

theorem smallest_k_for_64k_gt_4_20 : ∃ k : ℕ, k = 7 ∧ 64^k > 4^20 ∧ ∀ m : ℕ, m < k → 64^m ≤ 4^20 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_64k_gt_4_20_l3590_359010


namespace NUMINAMATH_CALUDE_plane_perpendicular_sufficient_not_necessary_l3590_359076

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (in_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_sufficient_not_necessary
  (α β : Plane) (m b c : Line) :
  intersect α β m →
  in_plane b α →
  in_plane c β →
  perpendicular c m →
  (plane_perpendicular α β → perpendicular c b) ∧
  ¬(perpendicular c b → plane_perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_sufficient_not_necessary_l3590_359076
