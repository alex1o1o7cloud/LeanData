import Mathlib

namespace NUMINAMATH_CALUDE_inverse_proportion_ordering_l2320_232093

/-- Represents a point on the inverse proportion function -/
structure InversePoint where
  x : ℝ
  y : ℝ
  k : ℝ
  h : y = k / x

/-- The theorem statement -/
theorem inverse_proportion_ordering
  (p₁ : InversePoint)
  (p₂ : InversePoint)
  (p₃ : InversePoint)
  (h₁ : p₁.x = -1)
  (h₂ : p₂.x = 2)
  (h₃ : p₃.x = 3)
  (hk : p₁.k = p₂.k ∧ p₂.k = p₃.k ∧ p₁.k < 0) :
  p₁.y > p₃.y ∧ p₃.y > p₂.y :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_ordering_l2320_232093


namespace NUMINAMATH_CALUDE_two_rooks_non_attacking_placements_l2320_232098

/-- The size of a standard chessboard --/
def boardSize : Nat := 8

/-- The total number of squares on the chessboard --/
def totalSquares : Nat := boardSize * boardSize

/-- The number of squares a rook can attack (excluding its own square) --/
def rookAttackSquares : Nat := 2 * boardSize - 1

/-- The number of ways to place two rooks on a chessboard without attacking each other --/
def twoRooksPlacement : Nat := totalSquares * (totalSquares - rookAttackSquares)

theorem two_rooks_non_attacking_placements :
  twoRooksPlacement = 3136 := by
  sorry

end NUMINAMATH_CALUDE_two_rooks_non_attacking_placements_l2320_232098


namespace NUMINAMATH_CALUDE_magnitude_a_minus_b_equals_5_l2320_232096

def vector_a : ℝ × ℝ := (-1, 1)
def vector_b : ℝ × ℝ := (3, -2)

theorem magnitude_a_minus_b_equals_5 :
  Real.sqrt ((vector_a.1 - vector_b.1)^2 + (vector_a.2 - vector_b.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_a_minus_b_equals_5_l2320_232096


namespace NUMINAMATH_CALUDE_cats_dogs_percentage_difference_l2320_232099

/-- Represents the number of animals in a compound -/
structure AnimalCount where
  cats : ℕ
  dogs : ℕ
  frogs : ℕ

/-- The conditions of the animal compound problem -/
def CompoundConditions (count : AnimalCount) : Prop :=
  count.cats < count.dogs ∧
  count.frogs = 2 * count.dogs ∧
  count.cats + count.dogs + count.frogs = 304 ∧
  count.frogs = 160

/-- The percentage difference between dogs and cats -/
def PercentageDifference (count : AnimalCount) : ℚ :=
  (count.dogs - count.cats : ℚ) / count.dogs * 100

/-- Theorem stating the percentage difference between dogs and cats -/
theorem cats_dogs_percentage_difference (count : AnimalCount) 
  (h : CompoundConditions count) : PercentageDifference count = 20 := by
  sorry


end NUMINAMATH_CALUDE_cats_dogs_percentage_difference_l2320_232099


namespace NUMINAMATH_CALUDE_greatest_power_of_seven_in_50_factorial_l2320_232015

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def highest_power_of_seven (n : ℕ) : ℕ :=
  if n < 7 then 0
  else (n / 7) + highest_power_of_seven (n / 7)

theorem greatest_power_of_seven_in_50_factorial :
  ∃ (z : ℕ), z = highest_power_of_seven 50 ∧
  (7^z : ℕ) ∣ factorial 50 ∧
  ∀ (y : ℕ), y > z → ¬((7^y : ℕ) ∣ factorial 50) :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_seven_in_50_factorial_l2320_232015


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2320_232019

-- Define the vectors
def a : ℝ × ℝ := (3, 1)
def b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem perpendicular_vectors_m_value :
  ∀ m : ℝ, dot_product a (b m) = 0 → m = -6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2320_232019


namespace NUMINAMATH_CALUDE_james_bike_ride_l2320_232088

/-- Proves that given the conditions of James' bike ride, the third hour distance is 25% farther than the second hour distance -/
theorem james_bike_ride (second_hour_distance : ℝ) (total_distance : ℝ) :
  second_hour_distance = 18 →
  second_hour_distance = (1 + 0.2) * (second_hour_distance / 1.2) →
  total_distance = 55.5 →
  (total_distance - (second_hour_distance + second_hour_distance / 1.2)) / second_hour_distance = 0.25 := by
  sorry

#check james_bike_ride

end NUMINAMATH_CALUDE_james_bike_ride_l2320_232088


namespace NUMINAMATH_CALUDE_jellybean_box_capacity_l2320_232023

theorem jellybean_box_capacity (bert_capacity : ℕ) (scale_factor : ℕ) : 
  bert_capacity = 150 → 
  scale_factor = 3 → 
  (scale_factor ^ 3 : ℕ) * bert_capacity = 4050 := by
sorry

end NUMINAMATH_CALUDE_jellybean_box_capacity_l2320_232023


namespace NUMINAMATH_CALUDE_no_polynomial_transform_l2320_232065

theorem no_polynomial_transform : ¬∃ (P : ℝ → ℝ), 
  (∀ x : ℝ, ∃ (a b c d : ℝ), P x = a * x^3 + b * x^2 + c * x + d) ∧
  P (-3) = -3 ∧ P (-1) = -1 ∧ P 1 = -3 ∧ P 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_no_polynomial_transform_l2320_232065


namespace NUMINAMATH_CALUDE_fruit_mix_kiwis_l2320_232097

theorem fruit_mix_kiwis (total : ℕ) (s b o k : ℕ) : 
  total = 340 →
  s + b + o + k = total →
  s = 3 * b →
  o = 2 * k →
  k = 5 * s →
  k = 104 := by
  sorry

end NUMINAMATH_CALUDE_fruit_mix_kiwis_l2320_232097


namespace NUMINAMATH_CALUDE_shower_tiles_width_l2320_232035

/-- Given a 3-walled shower with 20 tiles running the height of each wall and 480 tiles in total,
    the number of tiles running the width of each wall is 8. -/
theorem shower_tiles_width (num_walls : Nat) (height_tiles : Nat) (total_tiles : Nat) :
  num_walls = 3 → height_tiles = 20 → total_tiles = 480 →
  ∃ width_tiles : Nat, width_tiles = 8 ∧ num_walls * height_tiles * width_tiles = total_tiles :=
by sorry

end NUMINAMATH_CALUDE_shower_tiles_width_l2320_232035


namespace NUMINAMATH_CALUDE_parabola_focus_l2320_232026

/-- The parabola defined by the equation y = (1/4)x^2 -/
def parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (1/4) * p.1^2}

/-- The focus of a parabola is a point from which the distance to any point on the parabola
    is equal to the distance from that point to a fixed line called the directrix -/
def is_focus (f : ℝ × ℝ) (p : Set (ℝ × ℝ)) : Prop :=
  ∃ (d : ℝ), ∀ x y : ℝ, (x, y) ∈ p → 
    (x - f.1)^2 + (y - f.2)^2 = (y + d)^2

/-- The theorem stating that the focus of the parabola y = (1/4)x^2 is at (0, 1) -/
theorem parabola_focus :
  is_focus (0, 1) parabola := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l2320_232026


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l2320_232085

/-- The line equation -/
def line (a x y : ℝ) : Prop := (a + 1) * x + (3 * a - 1) * y - (6 * a + 2) = 0

/-- The ellipse equation -/
def ellipse (x y m : ℝ) : Prop := x^2 / 16 + y^2 / m = 1

/-- The theorem stating the conditions for the line and ellipse to always have a common point -/
theorem line_ellipse_intersection (a m : ℝ) :
  (∀ x y : ℝ, line a x y → ellipse x y m → False) ↔ 
  (m ∈ Set.Icc (16/7) 16 ∪ Set.Ioi 16) :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l2320_232085


namespace NUMINAMATH_CALUDE_bird_migration_difference_l2320_232002

/-- The number of bird families that flew away for the winter -/
def flew_away : ℕ := 86

/-- The number of bird families initially living near the mountain -/
def initial_families : ℕ := 45

/-- The difference between the number of bird families that flew away and those that stayed behind -/
def difference : ℕ := flew_away - initial_families

theorem bird_migration_difference :
  difference = 41 :=
sorry

end NUMINAMATH_CALUDE_bird_migration_difference_l2320_232002


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l2320_232009

theorem system_of_inequalities_solution :
  ∀ x y : ℤ,
    (2 * x - y > 3 ∧ 3 - 2 * x + y > 0) ↔ ((x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l2320_232009


namespace NUMINAMATH_CALUDE_carriage_hire_cost_l2320_232011

/-- The cost of hiring a carriage for a journey, given:
  * The distance to the destination
  * The speed of the horse
  * The hourly rate for the carriage
  * A flat fee for the service
-/
theorem carriage_hire_cost 
  (distance : ℝ) 
  (speed : ℝ) 
  (hourly_rate : ℝ) 
  (flat_fee : ℝ) 
  (h1 : distance = 20)
  (h2 : speed = 10)
  (h3 : hourly_rate = 30)
  (h4 : flat_fee = 20)
  : (distance / speed) * hourly_rate + flat_fee = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_carriage_hire_cost_l2320_232011


namespace NUMINAMATH_CALUDE_polygon_with_six_diagonals_has_nine_vertices_l2320_232081

/-- The number of vertices in a polygon given the number of diagonals from one vertex -/
def vertices_from_diagonals (diagonals : ℕ) : ℕ := diagonals + 3

/-- Theorem: A polygon with 6 diagonals drawn from one vertex has 9 vertices -/
theorem polygon_with_six_diagonals_has_nine_vertices :
  vertices_from_diagonals 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_six_diagonals_has_nine_vertices_l2320_232081


namespace NUMINAMATH_CALUDE_youngest_child_age_l2320_232054

def is_valid_age (x : ℕ) : Prop :=
  Nat.Prime x ∧
  Nat.Prime (x + 2) ∧
  Nat.Prime (x + 6) ∧
  Nat.Prime (x + 8) ∧
  Nat.Prime (x + 12) ∧
  Nat.Prime (x + 14)

theorem youngest_child_age :
  ∃ (x : ℕ), is_valid_age x ∧ ∀ (y : ℕ), y < x → ¬is_valid_age y :=
by sorry

end NUMINAMATH_CALUDE_youngest_child_age_l2320_232054


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2320_232060

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (square_diff : x^2 - y^2 = 48) : 
  |x - y| = 6 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2320_232060


namespace NUMINAMATH_CALUDE_triangle_properties_l2320_232074

theorem triangle_properties (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ratio : ∃ (k : ℝ), a = 2*k ∧ b = 5*k ∧ c = 6*k) 
  (h_area : (1/2) * a * c * Real.sqrt (1 - ((a^2 + c^2 - b^2) / (2*a*c))^2) = 3 * Real.sqrt 39 / 4) :
  ((a^2 + c^2 - b^2) / (2*a*c) = 5/8) ∧ (a + b + c = 13) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2320_232074


namespace NUMINAMATH_CALUDE_weekly_earnings_l2320_232016

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18
def tablet_repair_cost : ℕ := 12
def smartwatch_repair_cost : ℕ := 8

def phone_repairs : ℕ := 9
def laptop_repairs : ℕ := 5
def computer_repairs : ℕ := 4
def tablet_repairs : ℕ := 6
def smartwatch_repairs : ℕ := 8

def total_earnings : ℕ := 
  phone_repair_cost * phone_repairs +
  laptop_repair_cost * laptop_repairs +
  computer_repair_cost * computer_repairs +
  tablet_repair_cost * tablet_repairs +
  smartwatch_repair_cost * smartwatch_repairs

theorem weekly_earnings : total_earnings = 382 := by
  sorry

end NUMINAMATH_CALUDE_weekly_earnings_l2320_232016


namespace NUMINAMATH_CALUDE_roots_of_equation_number_of_roots_l2320_232039

def f (x : ℝ) : ℝ := x + |x^2 - 1|

theorem roots_of_equation (k : ℝ) :
  (∀ x, f x ≠ k) ∨
  (∃! x, f x = k) ∨
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = k ∧ f x₂ = k ∧ ∀ x, f x = k → x = x₁ ∨ x = x₂) ∨
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = k ∧ f x₂ = k ∧ f x₃ = k ∧
    ∀ x, f x = k → x = x₁ ∨ x = x₂ ∨ x = x₃) ∨
  (∃ x₁ x₂ x₃ x₄, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f x₁ = k ∧ f x₂ = k ∧ f x₃ = k ∧ f x₄ = k ∧
    ∀ x, f x = k → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) :=
by
  sorry

theorem number_of_roots (k : ℝ) :
  (k < -1 → ∀ x, f x ≠ k) ∧
  (k = -1 → ∃! x, f x = k) ∧
  (-1 < k ∧ k < 1 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = k ∧ f x₂ = k ∧ ∀ x, f x = k → x = x₁ ∨ x = x₂) ∧
  (k = 1 → ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = k ∧ f x₂ = k ∧ f x₃ = k ∧
    ∀ x, f x = k → x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (1 < k ∧ k < 5/4 → ∃ x₁ x₂ x₃ x₄, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f x₁ = k ∧ f x₂ = k ∧ f x₃ = k ∧ f x₄ = k ∧
    ∀ x, f x = k → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) ∧
  (k = 5/4 → ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = k ∧ f x₂ = k ∧ f x₃ = k ∧
    ∀ x, f x = k → x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (k > 5/4 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = k ∧ f x₂ = k ∧ ∀ x, f x = k → x = x₁ ∨ x = x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_number_of_roots_l2320_232039


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2320_232041

theorem trigonometric_identity (x y z : Real) 
  (hm : m = Real.sin x / Real.sin (y - z))
  (hn : n = Real.sin y / Real.sin (z - x))
  (hp : p = Real.sin z / Real.sin (x - y)) :
  m * n + n * p + p * m = -1 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2320_232041


namespace NUMINAMATH_CALUDE_max_pyramid_volume_l2320_232051

/-- The maximum volume of a pyramid SABC with given conditions -/
theorem max_pyramid_volume (AB AC : ℝ) (sin_BAC : ℝ) (h : ℝ) :
  AB = 5 →
  AC = 8 →
  sin_BAC = 4/5 →
  h ≤ (5 * Real.sqrt 137 * Real.sqrt 3) / 8 →
  (1/3 : ℝ) * (1/2 * AB * AC * sin_BAC) * h ≤ 10 * Real.sqrt (137/3) :=
by sorry

end NUMINAMATH_CALUDE_max_pyramid_volume_l2320_232051


namespace NUMINAMATH_CALUDE_complex_division_simplification_l2320_232053

theorem complex_division_simplification :
  let i : ℂ := Complex.I
  (2 * i) / (1 + i) = 1 + i :=
by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l2320_232053


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2320_232012

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a₃_eq : a 3 = -2
  aₙ_eq : ∃ n : ℕ, a n = 3/2
  Sₙ_eq : ∃ n : ℕ, (n : ℚ) * (a 1 + a n) / 2 = -15/2

/-- The first term of the arithmetic sequence is either -3 or -19/6 -/
theorem arithmetic_sequence_first_term (seq : ArithmeticSequence) :
  seq.a 1 = -3 ∨ seq.a 1 = -19/6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2320_232012


namespace NUMINAMATH_CALUDE_young_photographer_club_l2320_232004

theorem young_photographer_club (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ)
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  ∃ (mixed_groups : ℕ),
    mixed_groups = 72 ∧
    mixed_groups * 2 + boy_boy_photos + girl_girl_photos = total_groups * group_size :=
by sorry


end NUMINAMATH_CALUDE_young_photographer_club_l2320_232004


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2320_232083

theorem sufficient_not_necessary (a : ℝ) :
  (a > 1 → 1/a < 1) ∧ (∃ a, 1/a < 1 ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2320_232083


namespace NUMINAMATH_CALUDE_power_inequality_l2320_232050

theorem power_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  a^a < b^a := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2320_232050


namespace NUMINAMATH_CALUDE_probability_at_least_one_B_l2320_232075

/-- The probability of selecting at least one question of type B when randomly choosing 2 questions out of 5, where 2 are of type A and 3 are of type B -/
theorem probability_at_least_one_B (total : Nat) (type_A : Nat) (type_B : Nat) (select : Nat) : 
  total = 5 → type_A = 2 → type_B = 3 → select = 2 →
  (Nat.choose total select - Nat.choose type_A select) / Nat.choose total select = 9 / 10 := by
sorry


end NUMINAMATH_CALUDE_probability_at_least_one_B_l2320_232075


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2320_232090

theorem complex_fraction_simplification :
  (5 - Complex.I) / (1 - Complex.I) = 3 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2320_232090


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l2320_232006

/-- Given a boat's upstream and downstream travel times, 
    prove the ratio of current speed to boat speed in still water -/
theorem boat_speed_ratio 
  (distance : ℝ) 
  (upstream_time downstream_time : ℝ) 
  (h1 : distance = 15)
  (h2 : upstream_time = 5)
  (h3 : downstream_time = 3) :
  ∃ (boat_speed current_speed : ℝ),
    boat_speed > 0 ∧
    current_speed > 0 ∧
    distance / upstream_time = boat_speed - current_speed ∧
    distance / downstream_time = boat_speed + current_speed ∧
    current_speed / boat_speed = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l2320_232006


namespace NUMINAMATH_CALUDE_specific_det_value_det_equation_solution_l2320_232062

-- Define the determinant of order 2
def det2 (a b c d : ℤ) : ℤ := a * d - b * c

-- Theorem 1: The value of the specific determinant is 1
theorem specific_det_value : det2 2022 2023 2021 2022 = 1 := by sorry

-- Theorem 2: If the given determinant equals 32, then m = 4
theorem det_equation_solution (m : ℤ) : 
  det2 (m + 2) (m - 2) (m - 2) (m + 2) = 32 → m = 4 := by sorry

end NUMINAMATH_CALUDE_specific_det_value_det_equation_solution_l2320_232062


namespace NUMINAMATH_CALUDE_cake_mix_buyers_cake_mix_buyers_is_50_l2320_232022

theorem cake_mix_buyers (total_buyers : ℕ) (muffin_buyers : ℕ) (both_buyers : ℕ) 
  (neither_prob : ℚ) (h1 : total_buyers = 100) (h2 : muffin_buyers = 40) 
  (h3 : both_buyers = 15) (h4 : neither_prob = 1/4) : ℕ :=
by
  -- The number of buyers who purchase cake mix
  sorry

#check cake_mix_buyers

-- The theorem statement proves that given the conditions,
-- the number of buyers who purchase cake mix is 50
theorem cake_mix_buyers_is_50 : 
  cake_mix_buyers 100 40 15 (1/4) rfl rfl rfl rfl = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_cake_mix_buyers_cake_mix_buyers_is_50_l2320_232022


namespace NUMINAMATH_CALUDE_total_persimmons_l2320_232052

/-- Given that the total weight of persimmons is 3 kg and 5 persimmons weigh 1 kg,
    prove that the total number of persimmons is 15. -/
theorem total_persimmons (total_weight : ℝ) (weight_of_five : ℝ) (num_in_five : ℕ) :
  total_weight = 3 →
  weight_of_five = 1 →
  num_in_five = 5 →
  (total_weight / weight_of_five) * num_in_five = 15 := by
  sorry

#check total_persimmons

end NUMINAMATH_CALUDE_total_persimmons_l2320_232052


namespace NUMINAMATH_CALUDE_compare_expressions_l2320_232079

theorem compare_expressions : -|(-3/4)| < -(-4/5) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l2320_232079


namespace NUMINAMATH_CALUDE_quadrilateral_AD_length_l2320_232084

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Real × Real)

-- Define the conditions of the problem
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_BAC_eq_BDA (q : Quadrilateral) : Prop := sorry

def angle_BAD_eq_60 (q : Quadrilateral) : Prop := sorry

def angle_ADC_eq_60 (q : Quadrilateral) : Prop := sorry

def length_AB_eq_14 (q : Quadrilateral) : Real := sorry

def length_CD_eq_6 (q : Quadrilateral) : Real := sorry

def length_AD (q : Quadrilateral) : Real := sorry

-- Theorem statement
theorem quadrilateral_AD_length 
  (q : Quadrilateral) 
  (h1 : is_convex q)
  (h2 : angle_BAC_eq_BDA q)
  (h3 : angle_BAD_eq_60 q)
  (h4 : angle_ADC_eq_60 q)
  (h5 : length_AB_eq_14 q = 14)
  (h6 : length_CD_eq_6 q = 6) :
  length_AD q = 20 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_AD_length_l2320_232084


namespace NUMINAMATH_CALUDE_remainder_4059_div_32_l2320_232066

theorem remainder_4059_div_32 : 4059 % 32 = 27 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4059_div_32_l2320_232066


namespace NUMINAMATH_CALUDE_customers_without_tip_waiter_tip_problem_l2320_232040

theorem customers_without_tip (initial_customers : ℕ) (additional_customers : ℕ) (customers_with_tip : ℕ) : ℕ :=
  let total_customers := initial_customers + additional_customers
  total_customers - customers_with_tip

theorem waiter_tip_problem : customers_without_tip 29 20 15 = 34 := by
  sorry

end NUMINAMATH_CALUDE_customers_without_tip_waiter_tip_problem_l2320_232040


namespace NUMINAMATH_CALUDE_direct_proportion_m_value_l2320_232001

/-- A function f: ℝ → ℝ is a direct proportion if there exists a constant k such that f(x) = k * x for all x ∈ ℝ -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The given function -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ -7 * x + 2 + m

theorem direct_proportion_m_value :
  (∃ m : ℝ, is_direct_proportion (f m)) → (∃ m : ℝ, m = -2 ∧ is_direct_proportion (f m)) :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_m_value_l2320_232001


namespace NUMINAMATH_CALUDE_y_one_gt_y_two_l2320_232027

/-- Two points on a line with negative slope -/
structure PointsOnLine where
  y₁ : ℝ
  y₂ : ℝ
  h₁ : y₁ = -1/2 * (-5)
  h₂ : y₂ = -1/2 * (-2)

/-- Theorem: For two points A(-5, y₁) and B(-2, y₂) on the line y = -1/2x, y₁ > y₂ -/
theorem y_one_gt_y_two (p : PointsOnLine) : p.y₁ > p.y₂ := by
  sorry

end NUMINAMATH_CALUDE_y_one_gt_y_two_l2320_232027


namespace NUMINAMATH_CALUDE_equation_solution_l2320_232089

theorem equation_solution : ∃! x : ℝ, 13 + Real.sqrt (-4 + x - 3 * 3) = 14 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2320_232089


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2320_232008

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x + y) / (x - y) - (x - y) / (x + y) = 3) :
  (x^4 + y^4) / (x^4 - y^4) - (x^4 - y^4) / (x^4 + y^4) = 49 / 600 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2320_232008


namespace NUMINAMATH_CALUDE_correct_hours_calculation_l2320_232024

/-- Calculates the number of hours worked given hourly rates and total payment -/
def hours_worked (bricklayer_rate electrician_rate total_payment : ℚ) : ℚ :=
  total_payment / (bricklayer_rate + electrician_rate)

/-- Theorem stating that the calculated hours worked is correct -/
theorem correct_hours_calculation 
  (bricklayer_rate electrician_rate total_payment : ℚ) 
  (h1 : bricklayer_rate = 12)
  (h2 : electrician_rate = 16)
  (h3 : total_payment = 1350) :
  hours_worked bricklayer_rate electrician_rate total_payment = 1350 / 28 :=
by sorry

end NUMINAMATH_CALUDE_correct_hours_calculation_l2320_232024


namespace NUMINAMATH_CALUDE_inequalities_proof_l2320_232034

theorem inequalities_proof :
  (∀ x : ℝ, 3*x - 2*x^2 + 2 ≥ 0 ↔ 1/2 ≤ x ∧ x ≤ 2) ∧
  (∀ x : ℝ, 4 < |2*x - 3| ∧ |2*x - 3| ≤ 7 ↔ (5 ≥ x ∧ x > 7/2) ∨ (-2 ≤ x ∧ x < -1/2)) ∧
  (∀ x : ℝ, |x - 8| - |x - 4| > 2 ↔ x < 5) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2320_232034


namespace NUMINAMATH_CALUDE_base9_725_to_base3_l2320_232029

/-- Converts a base-9 digit to its two-digit base-3 representation -/
def base9_to_base3_digit (d : ℕ) : ℕ × ℕ :=
  (d / 3, d % 3)

/-- Converts a base-9 number to its base-3 representation -/
def base9_to_base3 (n : ℕ) : List ℕ :=
  let digits := n.digits 9
  List.join (digits.map (fun d => let (q, r) := base9_to_base3_digit d; [q, r]))

theorem base9_725_to_base3 :
  base9_to_base3 725 = [2, 1, 0, 2, 1, 2] := by
  sorry

end NUMINAMATH_CALUDE_base9_725_to_base3_l2320_232029


namespace NUMINAMATH_CALUDE_knowledge_contest_minimum_correct_answers_l2320_232091

theorem knowledge_contest_minimum_correct_answers
  (total_questions : ℕ)
  (correct_points : ℤ)
  (incorrect_points : ℤ)
  (minimum_score : ℤ)
  (h_total : total_questions = 20)
  (h_correct : correct_points = 10)
  (h_incorrect : incorrect_points = -3)
  (h_min_score : minimum_score = 70) :
  ∃ x : ℕ, x ≤ total_questions ∧
    correct_points * x + incorrect_points * (total_questions - x) ≥ minimum_score ∧
    ∀ y : ℕ, y < x →
      correct_points * y + incorrect_points * (total_questions - y) < minimum_score :=
by sorry

end NUMINAMATH_CALUDE_knowledge_contest_minimum_correct_answers_l2320_232091


namespace NUMINAMATH_CALUDE_max_integer_values_quadratic_l2320_232014

/-- Given a quadratic function f(x) = ax² + bx + c where a > 100,
    the maximum number of integer x values satisfying |f(x)| ≤ 50 is 2 -/
theorem max_integer_values_quadratic (a b c : ℝ) (ha : a > 100) :
  (∃ (n : ℕ), ∀ (S : Finset ℤ),
    (∀ x ∈ S, |a * x^2 + b * x + c| ≤ 50) →
    S.card ≤ n) ∧
  (∃ (S : Finset ℤ), (∀ x ∈ S, |a * x^2 + b * x + c| ≤ 50) ∧ S.card = 2) :=
sorry

end NUMINAMATH_CALUDE_max_integer_values_quadratic_l2320_232014


namespace NUMINAMATH_CALUDE_sum_of_digits_of_N_l2320_232086

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_of_digits_of_N (N : ℕ) (h : N^2 = 36^50 * 50^36) : sum_of_digits N = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_N_l2320_232086


namespace NUMINAMATH_CALUDE_track_width_l2320_232007

/-- Given two concentric circles where the outer circle has a circumference of 40π feet
    and the difference between the outer and inner circle circumferences is 16π feet,
    prove that the difference between their radii is 8 feet. -/
theorem track_width (r₁ r₂ : ℝ) : 
  (2 * π * r₁ = 40 * π) →  -- Outer circle circumference
  (2 * π * r₁ - 2 * π * r₂ = 16 * π) →  -- Difference in circumferences
  r₁ - r₂ = 8 := by sorry

end NUMINAMATH_CALUDE_track_width_l2320_232007


namespace NUMINAMATH_CALUDE_point_outside_circle_l2320_232003

theorem point_outside_circle 
  (a b : ℝ) 
  (line_intersects_circle : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ 
    a * x₁ + b * y₁ = 1 ∧ a * x₂ + b * y₂ = 1 ∧
    x₁^2 + y₁^2 = 1 ∧ x₂^2 + y₂^2 = 1) :
  a^2 + b^2 > 1 := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l2320_232003


namespace NUMINAMATH_CALUDE_max_stores_visited_is_three_l2320_232094

/-- Represents the shopping scenario in the town -/
structure ShoppingScenario where
  num_stores : ℕ
  total_visits : ℕ
  num_shoppers : ℕ
  two_store_visitors : ℕ

/-- The given shopping scenario -/
def given_scenario : ShoppingScenario :=
  { num_stores := 8
  , total_visits := 22
  , num_shoppers := 12
  , two_store_visitors := 8 }

/-- The maximum number of stores visited by any single person -/
def max_stores_visited (scenario : ShoppingScenario) : ℕ :=
  3

/-- Theorem stating that the maximum number of stores visited by any single person is 3 -/
theorem max_stores_visited_is_three (scenario : ShoppingScenario) 
  (h1 : scenario.num_stores = given_scenario.num_stores)
  (h2 : scenario.total_visits = given_scenario.total_visits)
  (h3 : scenario.num_shoppers = given_scenario.num_shoppers)
  (h4 : scenario.two_store_visitors = given_scenario.two_store_visitors)
  (h5 : scenario.two_store_visitors * 2 + (scenario.num_shoppers - scenario.two_store_visitors) ≤ scenario.total_visits)
  : max_stores_visited scenario = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_stores_visited_is_three_l2320_232094


namespace NUMINAMATH_CALUDE_min_y_value_of_trajectory_l2320_232049

/-- Given a 3D Cartesian coordinate system where:
    - Point A is at (3, 3, 0)
    - P(x, y, 0) is a moving point satisfying x = 2√((x-3)² + (y-3)²)
    Prove that the minimum value of y for all possible P is 3 - √3 -/
theorem min_y_value_of_trajectory (x y : ℝ) :
  x = 2 * Real.sqrt ((x - 3)^2 + (y - 3)^2) →
  ∃ (y_min : ℝ), y_min = 3 - Real.sqrt 3 ∧ ∀ y', x = 2 * Real.sqrt ((x - 3)^2 + (y' - 3)^2) → y' ≥ y_min :=
by sorry

end NUMINAMATH_CALUDE_min_y_value_of_trajectory_l2320_232049


namespace NUMINAMATH_CALUDE_queens_attack_probability_l2320_232005

/-- The size of the chessboard -/
def boardSize : Nat := 8

/-- The total number of squares on the chessboard -/
def totalSquares : Nat := boardSize * boardSize

/-- The number of ways to choose two different squares -/
def totalChoices : Nat := totalSquares * (totalSquares - 1) / 2

/-- The number of ways two queens can attack each other -/
def attackingChoices : Nat := 
  -- Same row
  boardSize * (boardSize * (boardSize - 1) / 2) +
  -- Same column
  boardSize * (boardSize * (boardSize - 1) / 2) +
  -- Same diagonal (main and anti-diagonals)
  (2 * (1 + 3 + 6 + 10 + 15 + 21) + 28)

/-- The probability of two queens attacking each other -/
def attackProbability : Rat := attackingChoices / totalChoices

theorem queens_attack_probability : 
  attackProbability = 7 / 24 := by sorry

end NUMINAMATH_CALUDE_queens_attack_probability_l2320_232005


namespace NUMINAMATH_CALUDE_sqrt_one_plus_xy_rational_l2320_232087

theorem sqrt_one_plus_xy_rational (x y : ℚ) 
  (h : (x^2 + y^2 - 2) * (x + y)^2 + (x*y + 1)^2 = 0) : 
  ∃ (q : ℚ), q^2 = 1 + x*y := by
  sorry

end NUMINAMATH_CALUDE_sqrt_one_plus_xy_rational_l2320_232087


namespace NUMINAMATH_CALUDE_stratified_sampling_total_employees_l2320_232043

/-- Given a stratified sampling of employees from four companies, 
    prove the total number of employees across all companies. -/
theorem stratified_sampling_total_employees 
  (total_A : ℕ) 
  (selected_A selected_B selected_C selected_D : ℕ) 
  (h1 : total_A = 96)
  (h2 : selected_A = 12)
  (h3 : selected_B = 21)
  (h4 : selected_C = 25)
  (h5 : selected_D = 43) :
  (total_A * (selected_A + selected_B + selected_C + selected_D)) / selected_A = 808 := by
  sorry

#check stratified_sampling_total_employees

end NUMINAMATH_CALUDE_stratified_sampling_total_employees_l2320_232043


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2320_232076

/-- Given a line with slope 5 passing through the point (-2, 4), 
    prove that the sum of its slope and y-intercept is 19. -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℝ), 
    m = 5 →                  -- The slope is 5
    4 = m * (-2) + b →       -- The line passes through (-2, 4)
    m + b = 19 :=            -- The sum of slope and y-intercept is 19
by
  sorry


end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2320_232076


namespace NUMINAMATH_CALUDE_sally_saturday_sandwiches_l2320_232057

/-- The number of sandwiches Sally eats on Saturday -/
def sandwiches_saturday : ℕ := 2

/-- The number of sandwiches Sally eats on Sunday -/
def sandwiches_sunday : ℕ := 1

/-- The number of pieces of bread used in each sandwich -/
def bread_per_sandwich : ℕ := 2

/-- The total number of pieces of bread Sally eats across Saturday and Sunday -/
def total_bread : ℕ := 6

/-- Theorem stating that Sally eats 2 sandwiches on Saturday -/
theorem sally_saturday_sandwiches :
  sandwiches_saturday = (total_bread - sandwiches_sunday * bread_per_sandwich) / bread_per_sandwich :=
by sorry

end NUMINAMATH_CALUDE_sally_saturday_sandwiches_l2320_232057


namespace NUMINAMATH_CALUDE_complex_expression_sum_l2320_232071

theorem complex_expression_sum (z : ℂ) : 
  z = Complex.exp (4 * Real.pi * I / 7) →
  z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_sum_l2320_232071


namespace NUMINAMATH_CALUDE_problem_statement_l2320_232017

theorem problem_statement (a b x y : ℝ) 
  (h1 : a + b = 2) 
  (h2 : x + y = 3) 
  (h3 : a * x + b * y = 4) : 
  (a^2 + b^2) * x * y + a * b * (x^2 + y^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2320_232017


namespace NUMINAMATH_CALUDE_line_x_axis_intersection_l2320_232010

theorem line_x_axis_intersection (x y : ℝ) :
  (5 * y - 7 * x = 14) ∧ (y = 0) → (x = -2 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_x_axis_intersection_l2320_232010


namespace NUMINAMATH_CALUDE_lcm_gcf_relations_l2320_232067

theorem lcm_gcf_relations :
  (∃! n : ℕ, Nat.lcm n 16 = 52 ∧ Nat.gcd n 16 = 8) ∧
  (¬ ∃ n : ℕ, Nat.lcm n 20 = 84 ∧ Nat.gcd n 20 = 4) ∧
  (∃! n : ℕ, Nat.lcm n 24 = 120 ∧ Nat.gcd n 24 = 6) := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_relations_l2320_232067


namespace NUMINAMATH_CALUDE_carpet_design_problem_l2320_232036

/-- Represents the dimensions of a rectangular region in the carpet design. -/
structure RegionDimensions where
  length : ℝ
  width : ℝ

/-- Represents the area of a region in the carpet design. -/
def area (d : RegionDimensions) : ℝ := d.length * d.width

/-- Checks if three real numbers form an arithmetic sequence. -/
def isArithmeticSequence (a b c : ℝ) : Prop := b - a = c - b

/-- The carpet design problem. -/
theorem carpet_design_problem (inner middle outer : RegionDimensions) 
    (h1 : inner.width = 2)
    (h2 : middle.width = inner.width + 4)
    (h3 : middle.length = inner.length + 4)
    (h4 : outer.width = middle.width + 4)
    (h5 : outer.length = middle.length + 4)
    (h6 : isArithmeticSequence (area inner) (area middle) (area outer)) :
    inner.length = 4 := by
  sorry

end NUMINAMATH_CALUDE_carpet_design_problem_l2320_232036


namespace NUMINAMATH_CALUDE_billy_sleep_theorem_l2320_232000

def night1_sleep : ℕ := 6

def night2_sleep : ℕ := night1_sleep + 2

def night3_sleep : ℕ := night2_sleep / 2

def night4_sleep : ℕ := night3_sleep * 3

def total_sleep : ℕ := night1_sleep + night2_sleep + night3_sleep + night4_sleep

theorem billy_sleep_theorem : total_sleep = 30 := by
  sorry

end NUMINAMATH_CALUDE_billy_sleep_theorem_l2320_232000


namespace NUMINAMATH_CALUDE_parallelogram_BJ_length_l2320_232077

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram ABCD with additional points H, J, and K -/
structure Parallelogram :=
  (A B C D H J K : Point)

/-- Checks if three points are collinear -/
def collinear (P Q R : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (P Q : Point) : ℝ := sorry

/-- Checks if two line segments are parallel -/
def parallel (P Q R S : Point) : Prop := sorry

theorem parallelogram_BJ_length
  (ABCD : Parallelogram)
  (h1 : collinear ABCD.A ABCD.D ABCD.H)
  (h2 : collinear ABCD.B ABCD.H ABCD.J)
  (h3 : collinear ABCD.B ABCD.H ABCD.K)
  (h4 : collinear ABCD.A ABCD.C ABCD.J)
  (h5 : collinear ABCD.D ABCD.C ABCD.K)
  (h6 : distance ABCD.J ABCD.H = 20)
  (h7 : distance ABCD.K ABCD.H = 30)
  (h8 : distance ABCD.A ABCD.D = 2 * distance ABCD.B ABCD.C)
  (h9 : parallel ABCD.A ABCD.B ABCD.D ABCD.C)
  (h10 : parallel ABCD.A ABCD.D ABCD.B ABCD.C) :
  distance ABCD.B ABCD.J = 5 := by sorry

end NUMINAMATH_CALUDE_parallelogram_BJ_length_l2320_232077


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2320_232082

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, 3*x - |(-2)*x + 1| ≥ a ↔ x ∈ Set.Ici 2) → a = 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2320_232082


namespace NUMINAMATH_CALUDE_sqrt_sum_diff_approx_l2320_232061

theorem sqrt_sum_diff_approx : 
  let x := Real.sqrt ((1 / 25) + (1 / 36) - (1 / 144))
  ∃ ε > 0, |x - 0.2467| < ε ∧ ε < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_diff_approx_l2320_232061


namespace NUMINAMATH_CALUDE_equation_solution_l2320_232064

theorem equation_solution : ∃ x : ℝ, 10.0003 * x = 10000.3 ∧ x = 1000 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2320_232064


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2320_232028

/-- Two numbers are inversely proportional if their product is constant -/
def inversely_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) :
  inversely_proportional x y →
  (∃ x₀ y₀ : ℝ, x₀ + y₀ = 60 ∧ x₀ = 3 * y₀ ∧ inversely_proportional x₀ y₀) →
  (x = -10 → y = -67.5) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2320_232028


namespace NUMINAMATH_CALUDE_solve_equation_l2320_232059

theorem solve_equation (x t : ℝ) : 
  (3 * (x + 5)) / 4 = t + (3 - 3 * x) / 2 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2320_232059


namespace NUMINAMATH_CALUDE_centroid_distance_theorem_l2320_232069

/-- Represents the possible distances from the centroid of a triangle to a plane -/
inductive CentroidDistance : Type
  | six : CentroidDistance
  | two : CentroidDistance
  | eight_thirds : CentroidDistance
  | four_thirds : CentroidDistance

/-- Given a triangle with vertices at distances 5, 6, and 7 from a plane,
    the distance from the centroid to the same plane is one of the defined values -/
theorem centroid_distance_theorem (d1 d2 d3 : ℝ) (h1 : d1 = 5) (h2 : d2 = 6) (h3 : d3 = 7) :
  ∃ (cd : CentroidDistance), true :=
sorry

end NUMINAMATH_CALUDE_centroid_distance_theorem_l2320_232069


namespace NUMINAMATH_CALUDE_teacher_distribution_count_l2320_232021

def distribute_teachers (n : ℕ) (k : ℕ) (min_a : ℕ) (min_others : ℕ) : ℕ :=
  -- n: total number of teachers
  -- k: number of schools
  -- min_a: minimum number of teachers for school A
  -- min_others: minimum number of teachers for other schools
  sorry

theorem teacher_distribution_count :
  distribute_teachers 6 4 2 1 = 660 := by sorry

end NUMINAMATH_CALUDE_teacher_distribution_count_l2320_232021


namespace NUMINAMATH_CALUDE_train_crossing_time_l2320_232031

theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ) 
  (h1 : train_length = 900)
  (h2 : platform_length = 1050)
  (h3 : platform_crossing_time = 39)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2320_232031


namespace NUMINAMATH_CALUDE_circle_area_when_radius_equals_six_times_reciprocal_of_circumference_l2320_232058

theorem circle_area_when_radius_equals_six_times_reciprocal_of_circumference :
  ∀ (r : ℝ), r > 0 → (6 * (1 / (2 * Real.pi * r)) = r) → (Real.pi * r^2 = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_when_radius_equals_six_times_reciprocal_of_circumference_l2320_232058


namespace NUMINAMATH_CALUDE_division_of_decimals_l2320_232048

theorem division_of_decimals : (0.08 : ℚ) / (0.002 : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_division_of_decimals_l2320_232048


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_6_l2320_232068

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem largest_four_digit_divisible_by_6 :
  ∀ n : ℕ, is_four_digit n → divisible_by_6 n → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_6_l2320_232068


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2320_232030

theorem chess_tournament_games (num_players : ℕ) (total_games : ℕ) (games_per_pair : ℕ) : 
  num_players = 8 →
  total_games = 56 →
  total_games = (num_players * (num_players - 1) * games_per_pair) / 2 →
  games_per_pair = 2 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2320_232030


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_p_and_q_l2320_232032

theorem not_p_sufficient_not_necessary_for_not_p_and_q (p q : Prop) :
  (∀ (h : ¬p), ¬(p ∧ q)) ∧
  ¬(∀ (h : ¬(p ∧ q)), ¬p) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_p_and_q_l2320_232032


namespace NUMINAMATH_CALUDE_same_color_probability_l2320_232080

/-- Represents the number of sides on each die -/
def totalSides : ℕ := 12

/-- Represents the number of red sides on each die -/
def redSides : ℕ := 3

/-- Represents the number of blue sides on each die -/
def blueSides : ℕ := 4

/-- Represents the number of green sides on each die -/
def greenSides : ℕ := 3

/-- Represents the number of purple sides on each die -/
def purpleSides : ℕ := 2

/-- Theorem stating the probability of rolling the same color on both dice -/
theorem same_color_probability : 
  (redSides^2 + blueSides^2 + greenSides^2 + purpleSides^2) / totalSides^2 = 19 / 72 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2320_232080


namespace NUMINAMATH_CALUDE_fraction_simplification_l2320_232038

theorem fraction_simplification (n : ℕ+) : (n : ℚ) * (3 : ℚ)^(n : ℕ) / (3 : ℚ)^(n : ℕ) = n := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2320_232038


namespace NUMINAMATH_CALUDE_calculation_sum_l2320_232046

theorem calculation_sum (x : ℝ) (h : (x - 5) + 14 = 39) : (5 * x + 14) + 39 = 203 := by
  sorry

end NUMINAMATH_CALUDE_calculation_sum_l2320_232046


namespace NUMINAMATH_CALUDE_horner_method_V3_l2320_232044

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + x^3 - x^2 + 3*x - 5

def horner_V3 (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  let a₅ := 1
  let a₄ := 2
  let a₃ := 1
  let a₂ := -1
  let a₁ := 3
  let a₀ := -5
  let V₀ := a₅
  let V₁ := V₀ * x + a₄
  let V₂ := V₁ * x + a₃
  V₂ * x + a₂

theorem horner_method_V3 :
  horner_V3 f 5 = 179 := by sorry

end NUMINAMATH_CALUDE_horner_method_V3_l2320_232044


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_to_both_red_l2320_232037

/-- Represents the color of a card -/
inductive Color
  | Red
  | Green
  | Blue

/-- Represents a pair of cards drawn from the bag -/
structure DrawnCards :=
  (first : Color)
  (second : Color)

/-- The bag containing 2 red, 2 green, and 2 blue cards -/
def bag : Multiset Color := 
  2 • {Color.Red} + 2 • {Color.Green} + 2 • {Color.Blue}

/-- Event: Both cards are red -/
def bothRed (draw : DrawnCards) : Prop :=
  draw.first = Color.Red ∧ draw.second = Color.Red

/-- Event: Neither of the 2 cards is red -/
def neitherRed (draw : DrawnCards) : Prop :=
  draw.first ≠ Color.Red ∧ draw.second ≠ Color.Red

/-- Event: Exactly one card is blue -/
def exactlyOneBlue (draw : DrawnCards) : Prop :=
  (draw.first = Color.Blue ∧ draw.second ≠ Color.Blue) ∨
  (draw.first ≠ Color.Blue ∧ draw.second = Color.Blue)

/-- Event: Both cards are green -/
def bothGreen (draw : DrawnCards) : Prop :=
  draw.first = Color.Green ∧ draw.second = Color.Green

theorem events_mutually_exclusive_to_both_red :
  ∀ (draw : DrawnCards),
    (bothRed draw → ¬(neitherRed draw)) ∧
    (bothRed draw → ¬(exactlyOneBlue draw)) ∧
    (bothRed draw → ¬(bothGreen draw)) :=
  sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_to_both_red_l2320_232037


namespace NUMINAMATH_CALUDE_range_of_power_function_l2320_232055

theorem range_of_power_function (m : ℝ) (h : m > 0) :
  Set.range (fun x : ℝ => x ^ m) ∩ Set.Ioo 0 1 = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_power_function_l2320_232055


namespace NUMINAMATH_CALUDE_expression_evaluation_l2320_232045

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  2 * (x^2 + 2*x*y) - 2*x^2 - x*y = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2320_232045


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2320_232095

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x - 1
  ∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2320_232095


namespace NUMINAMATH_CALUDE_largest_multiple_of_12_negation_greater_than_neg_150_l2320_232056

theorem largest_multiple_of_12_negation_greater_than_neg_150 :
  ∀ n : ℤ, 12 ∣ n ∧ -n > -150 → n ≤ 144 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_12_negation_greater_than_neg_150_l2320_232056


namespace NUMINAMATH_CALUDE_find_k_l2320_232092

theorem find_k (k : ℝ) (h1 : k ≠ 0) : 
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 2)) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2320_232092


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2320_232042

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2320_232042


namespace NUMINAMATH_CALUDE_segments_5_6_10_form_triangle_l2320_232073

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that the line segments 5, 6, and 10 can form a triangle. -/
theorem segments_5_6_10_form_triangle :
  can_form_triangle 5 6 10 := by sorry

end NUMINAMATH_CALUDE_segments_5_6_10_form_triangle_l2320_232073


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l2320_232020

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def rightmost_digit (n : ℕ) : ℕ := n % 10

def move_rightmost_to_leftmost (n : ℕ) : ℕ :=
  (n / 10) + (rightmost_digit n * 100000)

theorem unique_six_digit_number : 
  ∃! n : ℕ, is_six_digit n ∧ 
            rightmost_digit n = 2 ∧
            move_rightmost_to_leftmost n = 2 * n + 2 :=
by
  use 105262
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l2320_232020


namespace NUMINAMATH_CALUDE_probability_one_second_class_l2320_232033

def total_products : ℕ := 12
def first_class_products : ℕ := 10
def second_class_products : ℕ := 2
def selected_products : ℕ := 4

theorem probability_one_second_class :
  (Nat.choose second_class_products 1 * Nat.choose first_class_products (selected_products - 1)) /
  (Nat.choose total_products selected_products) = 16 / 33 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_second_class_l2320_232033


namespace NUMINAMATH_CALUDE_average_of_data_l2320_232013

def data : List ℝ := [4, 6, 5, 8, 7, 6]

theorem average_of_data :
  (data.sum / data.length : ℝ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_average_of_data_l2320_232013


namespace NUMINAMATH_CALUDE_maddie_thursday_viewing_l2320_232063

/-- Represents the viewing schedule for a TV show --/
structure ViewingSchedule where
  totalEpisodes : ℕ
  episodeLength : ℕ
  mondayMinutes : ℕ
  fridayEpisodes : ℕ
  weekendMinutes : ℕ

/-- Calculates the number of minutes watched on Thursday --/
def thursdayMinutes (schedule : ViewingSchedule) : ℕ :=
  schedule.totalEpisodes * schedule.episodeLength -
  (schedule.mondayMinutes + schedule.fridayEpisodes * schedule.episodeLength + schedule.weekendMinutes)

/-- Theorem stating that Maddie watched 21 minutes on Thursday --/
theorem maddie_thursday_viewing : 
  let schedule : ViewingSchedule := {
    totalEpisodes := 8,
    episodeLength := 44,
    mondayMinutes := 138,
    fridayEpisodes := 2,
    weekendMinutes := 105
  }
  thursdayMinutes schedule = 21 := by
  sorry

end NUMINAMATH_CALUDE_maddie_thursday_viewing_l2320_232063


namespace NUMINAMATH_CALUDE_gcd_of_consecutive_odd_terms_l2320_232072

theorem gcd_of_consecutive_odd_terms (n : ℕ) (h : Even n) (h_pos : 0 < n) :
  Nat.gcd ((n + 1) * (n + 3) * (n + 7) * (n + 9)) 15 = 15 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_consecutive_odd_terms_l2320_232072


namespace NUMINAMATH_CALUDE_solution_in_interval_l2320_232070

open Real

theorem solution_in_interval :
  ∃! x₀ : ℝ, 2 < x₀ ∧ x₀ < 3 ∧ Real.log x₀ + x₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l2320_232070


namespace NUMINAMATH_CALUDE_dodecahedral_die_expected_value_l2320_232018

/-- A fair dodecahedral die with faces numbered from 1 to 12 -/
def dodecahedral_die := Finset.range 12

/-- The probability of each outcome for a fair die -/
def prob (n : ℕ) : ℚ := 1 / 12

/-- The expected value of rolling the dodecahedral die -/
def expected_value : ℚ := (dodecahedral_die.sum fun i => (i + 1 : ℚ) * prob i) / 1

/-- Theorem: The expected value of rolling a fair dodecahedral die is 6.5 -/
theorem dodecahedral_die_expected_value : expected_value = 13/2 := by sorry

end NUMINAMATH_CALUDE_dodecahedral_die_expected_value_l2320_232018


namespace NUMINAMATH_CALUDE_probability_at_least_one_boy_l2320_232047

def total_students : ℕ := 5
def total_girls : ℕ := 3
def representatives : ℕ := 2

theorem probability_at_least_one_boy :
  let total_selections := Nat.choose total_students representatives
  let all_girl_selections := Nat.choose total_girls representatives
  (1 : ℚ) - (all_girl_selections : ℚ) / (total_selections : ℚ) = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_boy_l2320_232047


namespace NUMINAMATH_CALUDE_sum_first_150_remainder_l2320_232025

theorem sum_first_150_remainder (n : Nat) (divisor : Nat) : n = 150 → divisor = 11200 → 
  (n * (n + 1) / 2) % divisor = 125 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_150_remainder_l2320_232025


namespace NUMINAMATH_CALUDE_min_value_theorem_l2320_232078

theorem min_value_theorem (x : ℝ) (h : x > 0) : x^3 + 12*x + 81/x^4 ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2320_232078
