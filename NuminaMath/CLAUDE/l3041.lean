import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l3041_304160

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 1 / y) :
  (x * (1 / x)) * (y / (1 / y)) = 1 / x^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3041_304160


namespace NUMINAMATH_CALUDE_percentage_problem_l3041_304105

theorem percentage_problem (x : ℝ) :
  (15 / 100) * (30 / 100) * (50 / 100) * x = 108 → x = 4800 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3041_304105


namespace NUMINAMATH_CALUDE_power_seven_mod_twelve_l3041_304159

theorem power_seven_mod_twelve : 7^150 % 12 = 1 := by sorry

end NUMINAMATH_CALUDE_power_seven_mod_twelve_l3041_304159


namespace NUMINAMATH_CALUDE_tiles_for_dining_room_l3041_304148

/-- Calculates the number of tiles needed for a rectangular room with a border --/
def tiles_needed (room_length room_width border_width : ℕ) 
  (small_tile_size large_tile_size : ℕ) : ℕ :=
  let border_tiles := 
    2 * (2 * (room_length - 2 * border_width) + 2 * (room_width - 2 * border_width)) + 
    4 * border_width * border_width / (small_tile_size * small_tile_size)
  let inner_area := (room_length - 2 * border_width) * (room_width - 2 * border_width)
  let large_tiles := (inner_area + large_tile_size * large_tile_size - 1) / 
    (large_tile_size * large_tile_size)
  border_tiles + large_tiles

/-- Theorem stating that for the given room dimensions and tile sizes, 
    the total number of tiles needed is 144 --/
theorem tiles_for_dining_room : 
  tiles_needed 20 15 2 1 3 = 144 := by sorry

end NUMINAMATH_CALUDE_tiles_for_dining_room_l3041_304148


namespace NUMINAMATH_CALUDE_circle_problem_l3041_304171

theorem circle_problem (P : ℝ × ℝ) (S : ℝ × ℝ) (k : ℝ) :
  P = (5, 12) →
  S = (0, k) →
  (∃ (O : ℝ × ℝ), O = (0, 0) ∧
    ∃ (r₁ r₂ : ℝ), r₁ > 0 ∧ r₂ > 0 ∧
      (P.1 - O.1)^2 + (P.2 - O.2)^2 = r₁^2 ∧
      (S.1 - O.1)^2 + (S.2 - O.2)^2 = r₂^2 ∧
      r₁ - r₂ = 4) →
  k = 9 := by
sorry

end NUMINAMATH_CALUDE_circle_problem_l3041_304171


namespace NUMINAMATH_CALUDE_rectangle_width_length_ratio_l3041_304195

theorem rectangle_width_length_ratio 
  (w : ℝ) 
  (h1 : w > 0) 
  (h2 : 2 * (w + 10) = 30) : 
  w / 10 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_length_ratio_l3041_304195


namespace NUMINAMATH_CALUDE_specific_box_volume_l3041_304136

/-- The volume of an open box created from a rectangular sheet --/
def box_volume (sheet_length sheet_width x : ℝ) : ℝ :=
  (sheet_length - 2*x) * (sheet_width - 2*x) * x

/-- Theorem: The volume of the specific box is 4x^3 - 60x^2 + 216x --/
theorem specific_box_volume :
  ∀ x : ℝ, box_volume 18 12 x = 4*x^3 - 60*x^2 + 216*x :=
by
  sorry

end NUMINAMATH_CALUDE_specific_box_volume_l3041_304136


namespace NUMINAMATH_CALUDE_expression_value_at_two_l3041_304199

theorem expression_value_at_two :
  let x : ℕ := 2
  x + x * (x ^ x) = 10 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_two_l3041_304199


namespace NUMINAMATH_CALUDE_min_participants_is_61_l3041_304187

/-- Represents the number of participants in the race. -/
def n : ℕ := 61

/-- Represents the number of people who finished before Andrei. -/
def x : ℕ := 20

/-- Represents the number of people who finished before Dima. -/
def y : ℕ := 15

/-- Represents the number of people who finished before Lenya. -/
def z : ℕ := 12

/-- Theorem stating that 61 is the minimum number of participants satisfying the given conditions. -/
theorem min_participants_is_61 :
  (x + 1 + 2 * x = n) ∧
  (y + 1 + 3 * y = n) ∧
  (z + 1 + 4 * z = n) ∧
  (∀ m : ℕ, m < n → ¬((m - 1) % 3 = 0 ∧ (m - 1) % 4 = 0 ∧ (m - 1) % 5 = 0)) :=
by sorry

#check min_participants_is_61

end NUMINAMATH_CALUDE_min_participants_is_61_l3041_304187


namespace NUMINAMATH_CALUDE_convertible_count_l3041_304100

theorem convertible_count (total : ℕ) (regular_percent : ℚ) (truck_percent : ℚ) :
  total = 125 →
  regular_percent = 64 / 100 →
  truck_percent = 8 / 100 →
  (total : ℚ) * regular_percent + (total : ℚ) * truck_percent + 35 = total :=
by sorry

end NUMINAMATH_CALUDE_convertible_count_l3041_304100


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l3041_304123

theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 30 * S) :
  (S - C) / C * 100 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l3041_304123


namespace NUMINAMATH_CALUDE_m_range_characterization_l3041_304175

/-- Proposition p: For all x ∈ ℝ, x² + 2x > m -/
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x > m

/-- Proposition q: There exists x₀ ∈ ℝ, such that x₀² + 2mx₀ + 2 - m ≤ 0 -/
def q (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 2 - m ≤ 0

/-- The range of values for m -/
def m_range (m : ℝ) : Prop := (m > -2 ∧ m < -1) ∨ m ≥ 1

theorem m_range_characterization (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m :=
sorry

end NUMINAMATH_CALUDE_m_range_characterization_l3041_304175


namespace NUMINAMATH_CALUDE_circle_radius_with_chords_l3041_304156

/-- A circle with three parallel chords -/
structure CircleWithChords where
  -- The radius of the circle
  radius : ℝ
  -- The distance from the center to the closest chord
  x : ℝ
  -- The common distance between the chords
  y : ℝ
  -- Conditions on the chords
  chord_condition : radius^2 = x^2 + 100 ∧ 
                    radius^2 = (x + y)^2 + 64 ∧ 
                    radius^2 = (x + 2*y)^2 + 16

/-- The theorem stating that the radius of the circle with the given chord configuration is 5√22/2 -/
theorem circle_radius_with_chords (c : CircleWithChords) : c.radius = 5 * Real.sqrt 22 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_with_chords_l3041_304156


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3041_304132

theorem quadratic_roots_range (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + (3*a - 1)*x + a + 8 = 0 ↔ x = x₁ ∨ x = x₂) →  -- quadratic equation with roots x₁ and x₂
  x₁ ≠ x₂ →  -- distinct roots
  x₁ < 1 →   -- x₁ < 1
  x₂ > 1 →   -- x₂ > 1
  a < -2 :=  -- range of a
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3041_304132


namespace NUMINAMATH_CALUDE_parallelogram_height_l3041_304150

/-- Given a parallelogram with sides 20 feet and 60 feet, and height 55 feet
    perpendicular to the 20-foot side, prove that the height perpendicular
    to the 60-foot side is 1100/60 feet. -/
theorem parallelogram_height (a b h : ℝ) (ha : a = 20) (hb : b = 60) (hh : h = 55) :
  a * h / b = 1100 / 60 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3041_304150


namespace NUMINAMATH_CALUDE_area_triangle_AOB_l3041_304177

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line intersecting a parabola -/
structure IntersectingLine (p : Parabola) where
  pointA : ℝ × ℝ
  pointB : ℝ × ℝ
  passes_through_focus : (pointA.1 - p.focus.1) * (pointB.2 - p.focus.2) = 
                         (pointB.1 - p.focus.1) * (pointA.2 - p.focus.2)

/-- Theorem: Area of triangle AOB for a specific parabola and intersecting line -/
theorem area_triangle_AOB 
  (p : Parabola) 
  (l : IntersectingLine p) 
  (h_parabola : p.equation = fun x y => y^2 = 4*x) 
  (h_focus : p.focus = (1, 0)) 
  (h_AF_length : Real.sqrt ((l.pointA.1 - p.focus.1)^2 + (l.pointA.2 - p.focus.2)^2) = 3) :
  let O : ℝ × ℝ := (0, 0)
  Real.sqrt (
    (l.pointA.1 * l.pointB.2 - l.pointB.1 * l.pointA.2)^2 +
    (l.pointA.1 * O.2 - O.1 * l.pointA.2)^2 +
    (O.1 * l.pointB.2 - l.pointB.1 * O.2)^2
  ) / 2 = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_AOB_l3041_304177


namespace NUMINAMATH_CALUDE_cloth_cost_per_meter_l3041_304158

theorem cloth_cost_per_meter (total_length : ℝ) (total_cost : ℝ) 
  (h1 : total_length = 9.25)
  (h2 : total_cost = 434.75) :
  total_cost / total_length = 47 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_per_meter_l3041_304158


namespace NUMINAMATH_CALUDE_empty_set_proof_l3041_304173

theorem empty_set_proof : {x : ℝ | x^2 - x + 1 = 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_empty_set_proof_l3041_304173


namespace NUMINAMATH_CALUDE_max_value_z_minus_i_l3041_304141

theorem max_value_z_minus_i (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (M : ℝ), M = 3 ∧ ∀ w, Complex.abs w = 2 → Complex.abs (w - Complex.I) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_z_minus_i_l3041_304141


namespace NUMINAMATH_CALUDE_fifth_term_is_two_l3041_304145

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying specific conditions, prove that its fifth term is 2. -/
theorem fifth_term_is_two (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_2 : a 2 = 2 * a 3 + 1)
  (h_4 : a 4 = 2 * a 3 + 7) : 
  a 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_two_l3041_304145


namespace NUMINAMATH_CALUDE_simultaneous_divisibility_l3041_304185

theorem simultaneous_divisibility (x y : ℤ) :
  (17 ∣ (2 * x + 3 * y)) ↔ (17 ∣ (9 * x + 5 * y)) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_divisibility_l3041_304185


namespace NUMINAMATH_CALUDE_expression_simplification_l3041_304198

theorem expression_simplification :
  let x := Real.pi / 18  -- 10 degrees in radians
  (Real.sqrt (1 - 2 * Real.sin x * Real.cos x)) /
  (Real.sin (17 * x) - Real.sqrt (1 - Real.sin (17 * x) ^ 2)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3041_304198


namespace NUMINAMATH_CALUDE_john_vacation_expenses_l3041_304128

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Calculates the remaining money after buying a ticket -/
def remaining_money (savings : ℕ) (ticket_cost : ℕ) : ℕ :=
  base8_to_base10 savings - ticket_cost

theorem john_vacation_expenses :
  remaining_money 5373 1500 = 1311 := by sorry

end NUMINAMATH_CALUDE_john_vacation_expenses_l3041_304128


namespace NUMINAMATH_CALUDE_probability_not_ab_l3041_304118

def num_courses : ℕ := 4
def num_selected : ℕ := 2

def probability_not_selected_together : ℚ :=
  1 - (1 / (num_courses.choose num_selected))

theorem probability_not_ab : probability_not_selected_together = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_ab_l3041_304118


namespace NUMINAMATH_CALUDE_max_students_distribution_l3041_304119

theorem max_students_distribution (pens pencils erasers notebooks : ℕ) 
  (h_pens : pens = 4261)
  (h_pencils : pencils = 2677)
  (h_erasers : erasers = 1759)
  (h_notebooks : notebooks = 1423) :
  (∃ (n : ℕ), n > 0 ∧ 
    pens % n = 0 ∧ pencils % n = 0 ∧ erasers % n = 0 ∧ notebooks % n = 0 ∧
    (∀ m : ℕ, m > n → (pens % m ≠ 0 ∨ pencils % m ≠ 0 ∨ erasers % m ≠ 0 ∨ notebooks % m ≠ 0))) →
  1 = Nat.gcd (Nat.gcd (Nat.gcd pens pencils) erasers) notebooks :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l3041_304119


namespace NUMINAMATH_CALUDE_infinite_circles_inside_l3041_304102

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define what it means for a point to be inside a circle
def isInside (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

-- Define what it means for a circle to be entirely inside another circle
def isEntirelyInside (c1 c2 : Circle) : Prop :=
  ∀ p : Point, isInside p c1 → isInside p c2

-- The main theorem
theorem infinite_circles_inside (C : Circle) (A B : Point) 
  (hA : isInside A C) (hB : isInside B C) :
  ∃ f : ℕ → Circle, (∀ n : ℕ, isEntirelyInside (f n) C ∧ 
                               isInside A (f n) ∧ 
                               isInside B (f n)) ∧
                     (∀ m n : ℕ, m ≠ n → f m ≠ f n) := by
  sorry

end NUMINAMATH_CALUDE_infinite_circles_inside_l3041_304102


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l3041_304164

/-- An isosceles right triangle with perimeter 8 + 8√2 has a hypotenuse of length 8 -/
theorem isosceles_right_triangle_hypotenuse (a c : ℝ) : 
  a > 0 → -- side length is positive
  c > 0 → -- hypotenuse length is positive
  a + a + c = 8 + 8 * Real.sqrt 2 → -- perimeter condition
  a * a + a * a = c * c → -- Pythagorean theorem for isosceles right triangle
  c = 8 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l3041_304164


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l3041_304143

/-- Given a circle C with equation x^2 + 8x + y^2 - 2y = -4, 
    prove that u + v + s = -3 + √13, where (u,v) is the center and s is the radius -/
theorem circle_center_radius_sum (x y : ℝ) : 
  (∃ (u v s : ℝ), x^2 + 8*x + y^2 - 2*y = -4 ∧ 
  (x - u)^2 + (y - v)^2 = s^2) → 
  (∃ (u v s : ℝ), x^2 + 8*x + y^2 - 2*y = -4 ∧ 
  (x - u)^2 + (y - v)^2 = s^2 ∧ 
  u + v + s = -3 + Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l3041_304143


namespace NUMINAMATH_CALUDE_gcd_204_85_l3041_304126

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l3041_304126


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_cubed_l3041_304174

theorem imaginary_part_of_one_minus_i_cubed (i : ℂ) : 
  i^2 = -1 → Complex.im ((1 - i)^3) = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_cubed_l3041_304174


namespace NUMINAMATH_CALUDE_fifteenth_base_five_number_l3041_304104

/-- Represents a number in base 5 --/
def BaseFive : Type := Nat

/-- Converts a natural number to its base 5 representation --/
def toBaseFive (n : Nat) : BaseFive :=
  sorry

/-- The sequence of numbers in base 5 --/
def baseFiveSequence : Nat → BaseFive :=
  sorry

theorem fifteenth_base_five_number :
  baseFiveSequence 15 = toBaseFive 30 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_base_five_number_l3041_304104


namespace NUMINAMATH_CALUDE_fayes_math_problems_l3041_304168

theorem fayes_math_problems :
  ∀ (total_problems math_problems science_problems finished_problems remaining_problems : ℕ),
    science_problems = 9 →
    finished_problems = 40 →
    remaining_problems = 15 →
    total_problems = math_problems + science_problems →
    total_problems = finished_problems + remaining_problems →
    math_problems = 46 := by
  sorry

end NUMINAMATH_CALUDE_fayes_math_problems_l3041_304168


namespace NUMINAMATH_CALUDE_compound_oxygen_count_l3041_304193

/-- Represents the number of atoms of a particular element in a compound -/
structure AtomCount where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given its atom counts and atomic weights -/
def molecularWeight (atoms : AtomCount) (carbonWeight hydrogenWeight oxygenWeight : ℝ) : ℝ :=
  atoms.carbon * carbonWeight + atoms.hydrogen * hydrogenWeight + atoms.oxygen * oxygenWeight

/-- Theorem stating that a compound with formula C3H6 and molecular weight 58 g/mol contains 1 oxygen atom -/
theorem compound_oxygen_count : 
  ∀ (atoms : AtomCount),
    atoms.carbon = 3 →
    atoms.hydrogen = 6 →
    molecularWeight atoms 12.01 1.008 16.00 = 58 →
    atoms.oxygen = 1 := by
  sorry

end NUMINAMATH_CALUDE_compound_oxygen_count_l3041_304193


namespace NUMINAMATH_CALUDE_larger_number_problem_l3041_304140

theorem larger_number_problem (x y : ℕ) 
  (h1 : y - x = 1500)
  (h2 : y = 6 * x + 15) : 
  y = 1797 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3041_304140


namespace NUMINAMATH_CALUDE_gcd_of_840_and_1764_l3041_304113

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_840_and_1764_l3041_304113


namespace NUMINAMATH_CALUDE_pump_fill_time_proof_l3041_304149

/-- The time it takes to fill the tank with the leak (in hours) -/
def fill_time_with_leak : ℝ := 20

/-- The time it takes for the leak to empty the tank (in hours) -/
def leak_empty_time : ℝ := 5

/-- The time it takes for the pump to fill the tank without the leak (in hours) -/
def pump_fill_time : ℝ := 4

theorem pump_fill_time_proof : 
  (1 / pump_fill_time - 1 / leak_empty_time) * fill_time_with_leak = 1 := by
  sorry

end NUMINAMATH_CALUDE_pump_fill_time_proof_l3041_304149


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l3041_304184

def in_quadrant_I_or_II (x y : ℝ) : Prop := (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)

theorem points_in_quadrants_I_and_II (x y : ℝ) :
  y > 3 * x → y > 6 - x → in_quadrant_I_or_II x y := by
  sorry

end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l3041_304184


namespace NUMINAMATH_CALUDE_max_gross_profit_l3041_304131

/-- Represents the daily sales volume as a function of selling price -/
def sales_volume (x : ℝ) : ℝ := -x + 26

/-- Represents the gross profit as a function of selling price -/
def gross_profit (x : ℝ) : ℝ := x * (sales_volume x) - 4 * (sales_volume x)

/-- Theorem stating the maximum gross profit under given constraints -/
theorem max_gross_profit :
  ∃ (max_profit : ℝ),
    (∀ x : ℝ, 6 ≤ x ∧ x ≤ 12 ∧ sales_volume x ≤ 10 → gross_profit x ≤ max_profit) ∧
    (∃ x : ℝ, 6 ≤ x ∧ x ≤ 12 ∧ sales_volume x ≤ 10 ∧ gross_profit x = max_profit) ∧
    max_profit = 120 := by
  sorry

end NUMINAMATH_CALUDE_max_gross_profit_l3041_304131


namespace NUMINAMATH_CALUDE_birds_to_asia_count_l3041_304144

/-- The number of bird families that flew to Asia -/
def birds_to_asia (initial : ℕ) (to_africa : ℕ) (remaining : ℕ) : ℕ :=
  initial - to_africa - remaining

/-- Theorem stating that 37 bird families flew to Asia -/
theorem birds_to_asia_count : birds_to_asia 85 23 25 = 37 := by
  sorry

end NUMINAMATH_CALUDE_birds_to_asia_count_l3041_304144


namespace NUMINAMATH_CALUDE_sangita_flying_months_l3041_304129

/-- Calculates the number of months needed to complete flying hours for a pilot certificate. -/
def months_to_complete_flying (total_required : ℕ) (day_completed : ℕ) (night_completed : ℕ) (cross_country_completed : ℕ) (monthly_goal : ℕ) : ℕ :=
  let total_completed := day_completed + night_completed + cross_country_completed
  let remaining_hours := total_required - total_completed
  (remaining_hours + monthly_goal - 1) / monthly_goal

/-- Proves that Sangita needs 6 months to complete her flying hours. -/
theorem sangita_flying_months :
  months_to_complete_flying 1500 50 9 121 220 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sangita_flying_months_l3041_304129


namespace NUMINAMATH_CALUDE_min_value_of_some_expression_l3041_304116

-- Define the expression as a function of x and some_expression
def f (x : ℝ) (some_expression : ℝ) : ℝ :=
  |x - 4| + |x + 6| + |some_expression|

-- State the theorem
theorem min_value_of_some_expression :
  (∃ (some_expression : ℝ), ∀ (x : ℝ), f x some_expression ≥ 11) →
  (∃ (some_expression : ℝ), (∀ (x : ℝ), f x some_expression ≥ 11) ∧ |some_expression| = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_some_expression_l3041_304116


namespace NUMINAMATH_CALUDE_rates_sum_of_squares_l3041_304153

/-- Represents the rates of biking, jogging, and swimming -/
structure Rates where
  biking : ℕ
  jogging : ℕ
  swimming : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (r : Rates) : Prop :=
  3 * r.biking + 2 * r.jogging + 4 * r.swimming = 84 ∧
  4 * r.biking + 3 * r.jogging + 2 * r.swimming = 106

/-- The theorem to be proved -/
theorem rates_sum_of_squares (r : Rates) : 
  satisfies_conditions r → r.biking^2 + r.jogging^2 + r.swimming^2 = 1125 := by
  sorry


end NUMINAMATH_CALUDE_rates_sum_of_squares_l3041_304153


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l3041_304192

/-- Quadratic function passing through specific points with given minimum --/
theorem quadratic_function_proof (a h k : ℝ) :
  a ≠ 0 →
  a * (1 - h)^2 + k = 3 →
  a * (3 - h)^2 + k = 3 →
  k = -1 →
  a = 4 ∧ h = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l3041_304192


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l3041_304137

theorem missing_fraction_sum (x : ℚ) :
  1/3 + 1/2 + 1/5 + 1/4 + (-9/20) + (-9/20) + x = 9/20 →
  x = 1/15 := by
sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l3041_304137


namespace NUMINAMATH_CALUDE_polynomial_roots_sum_l3041_304167

theorem polynomial_roots_sum (c d : ℝ) : 
  c^2 - 6*c + 10 = 0 ∧ d^2 - 6*d + 10 = 0 → c^3 + c^5*d^3 + c^3*d^5 + d^3 = 16156 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_sum_l3041_304167


namespace NUMINAMATH_CALUDE_vector_sum_parallel_l3041_304101

/-- Given two parallel vectors a and b in R², prove that their linear combination results in (-4, -8) -/
theorem vector_sum_parallel (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, m]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  (2 • a + 3 • b : Fin 2 → ℝ) = ![-4, -8] := by
sorry

end NUMINAMATH_CALUDE_vector_sum_parallel_l3041_304101


namespace NUMINAMATH_CALUDE_inequality_implies_range_l3041_304133

/-- The inequality condition for all x > 1 -/
def inequality_condition (a : ℝ) : Prop :=
  ∀ x > 1, a * (x - 1) ≥ Real.log (x - 1)

/-- The range of a satisfying the inequality condition -/
def a_range (a : ℝ) : Prop :=
  a ≥ 1 / Real.exp 1

theorem inequality_implies_range :
  ∀ a : ℝ, inequality_condition a → a_range a :=
by sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l3041_304133


namespace NUMINAMATH_CALUDE_range_of_m_l3041_304196

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (|x - 3| ≤ 2 → (x - m + 1) * (x - m - 1) ≤ 0) ∧ 
   (∃ y : ℝ, (y - m + 1) * (y - m - 1) ≤ 0 ∧ |y - 3| > 2)) →
  2 ≤ m ∧ m ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3041_304196


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_17_l3041_304197

theorem consecutive_integers_around_sqrt_17 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 17) → (Real.sqrt 17 < b) → (a + b = 9) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_17_l3041_304197


namespace NUMINAMATH_CALUDE_union_and_intersection_range_of_a_l3041_304162

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 5 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | a * x + 1 > 0}

-- Theorem for part (I)
theorem union_and_intersection :
  (A ∪ B = {x | 1 ≤ x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | 6 ≤ x ∧ x < 10}) := by sorry

-- Theorem for part (II)
theorem range_of_a :
  ∀ a : ℝ, (A ∩ C a = A) → a ∈ Set.Ici (-1/6) := by sorry

end NUMINAMATH_CALUDE_union_and_intersection_range_of_a_l3041_304162


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3041_304169

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 25 →           -- Hypotenuse is 25
  a ≤ b →            -- a is the shorter leg
  (a = 7 ∨ a = 20) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3041_304169


namespace NUMINAMATH_CALUDE_correct_observation_value_l3041_304163

/-- Proves that the correct value of a misrecorded observation is 48, given the conditions of the problem. -/
theorem correct_observation_value (n : ℕ) (original_mean corrected_mean wrong_value : ℚ)
  (h_n : n = 50)
  (h_original_mean : original_mean = 32)
  (h_corrected_mean : corrected_mean = 32.5)
  (h_wrong_value : wrong_value = 23) :
  let correct_value := (n : ℚ) * corrected_mean - ((n : ℚ) * original_mean - wrong_value)
  correct_value = 48 := by
  sorry

end NUMINAMATH_CALUDE_correct_observation_value_l3041_304163


namespace NUMINAMATH_CALUDE_expression_value_at_three_l3041_304111

theorem expression_value_at_three : 
  let x : ℝ := 3
  x^6 - x^3 - 6*x = 684 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l3041_304111


namespace NUMINAMATH_CALUDE_power_division_23_l3041_304180

theorem power_division_23 : (23 ^ 11 : ℕ) / (23 ^ 5) = 148035889 := by sorry

end NUMINAMATH_CALUDE_power_division_23_l3041_304180


namespace NUMINAMATH_CALUDE_triangle_side_length_l3041_304155

/-- Given a triangle ABC with the specified properties, prove that AC = 5√3 -/
theorem triangle_side_length (A B C : ℝ) (BC : ℝ) :
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  (0 < C) ∧ (C < π) →
  A + B + C = π →
  2 * Real.sin (A - B) + Real.cos (B + C) = 2 →
  BC = 5 →
  ∃ (AC : ℝ), AC = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3041_304155


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3041_304147

theorem quadratic_roots_sum (m n : ℝ) : 
  (m^2 + 2*m - 2022 = 0) → 
  (n^2 + 2*n - 2022 = 0) → 
  (m^2 + 3*m + n = 2020) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3041_304147


namespace NUMINAMATH_CALUDE_rectangle_area_l3041_304161

theorem rectangle_area (x y : ℝ) 
  (h_perimeter : x + y = 5)
  (h_diagonal : x^2 + y^2 = 15) : 
  x * y = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3041_304161


namespace NUMINAMATH_CALUDE_number_with_specific_remainders_l3041_304183

theorem number_with_specific_remainders : ∃ (N : ℕ), N % 13 = 11 ∧ N % 17 = 9 ∧ N = 141 := by
  sorry

end NUMINAMATH_CALUDE_number_with_specific_remainders_l3041_304183


namespace NUMINAMATH_CALUDE_rectangular_to_polar_l3041_304157

theorem rectangular_to_polar :
  let x : ℝ := 2 * Real.sqrt 3
  let y : ℝ := -2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := if x > 0 ∧ y < 0 then 2 * π + Real.arctan (y / x) else Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ r = 4 ∧ θ = 11 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_l3041_304157


namespace NUMINAMATH_CALUDE_train_length_l3041_304170

/-- Given a train with speed 50 km/hr crossing a pole in 9 seconds, its length is 125 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 50 → -- speed in km/hr
  time = 9 → -- time in seconds
  length = speed * (1000 / 3600) * time → -- length calculation
  length = 125 := by sorry

end NUMINAMATH_CALUDE_train_length_l3041_304170


namespace NUMINAMATH_CALUDE_gcf_of_60_90_150_l3041_304124

theorem gcf_of_60_90_150 : Nat.gcd 60 (Nat.gcd 90 150) = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_60_90_150_l3041_304124


namespace NUMINAMATH_CALUDE_exchange_result_l3041_304190

/-- The exchange rate from Canadian dollars (CAD) to Japanese yen (JPY) -/
def exchange_rate : ℝ := 85

/-- The amount of Canadian dollars to be exchanged -/
def cad_amount : ℝ := 5

/-- Theorem stating that exchanging 5 CAD results in 425 JPY -/
theorem exchange_result : cad_amount * exchange_rate = 425 := by
  sorry

end NUMINAMATH_CALUDE_exchange_result_l3041_304190


namespace NUMINAMATH_CALUDE_unique_number_exists_l3041_304106

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    a + b + c = 10 ∧
    b = a + c ∧
    100 * c + 10 * b + a = 100 * a + 10 * b + c + 99

theorem unique_number_exists :
  ∃! n : ℕ, is_valid_number n ∧ n = 203 :=
sorry

end NUMINAMATH_CALUDE_unique_number_exists_l3041_304106


namespace NUMINAMATH_CALUDE_number_puzzle_l3041_304172

theorem number_puzzle : ∃! x : ℝ, x - 18 = 3 * (86 - x) := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l3041_304172


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3041_304188

/-- 
Theorem: In a rhombus with side length 65 units and shorter diagonal 56 units, 
the longer diagonal is 118 units.
-/
theorem rhombus_longer_diagonal 
  (side_length : ℝ) 
  (shorter_diagonal : ℝ) 
  (h1 : side_length = 65) 
  (h2 : shorter_diagonal = 56) : ℝ :=
by
  -- Define the longer diagonal
  let longer_diagonal : ℝ := 118
  
  -- The proof would go here
  sorry

#check rhombus_longer_diagonal

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3041_304188


namespace NUMINAMATH_CALUDE_total_points_earned_l3041_304166

/-- The number of pounds required to earn one point -/
def pounds_per_point : ℕ := 4

/-- The number of pounds Paige recycled -/
def paige_pounds : ℕ := 14

/-- The number of pounds Paige's friends recycled -/
def friends_pounds : ℕ := 2

/-- The total number of pounds recycled -/
def total_pounds : ℕ := paige_pounds + friends_pounds

/-- The theorem stating that the total points earned is 4 -/
theorem total_points_earned : (total_pounds / pounds_per_point : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_points_earned_l3041_304166


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_difference_of_roots_x2_minus_7x_plus_9_l3041_304154

theorem quadratic_roots_difference (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 →
  |x₁ - x₂| = Real.sqrt ((b^2 - 4*a*c) / a^2) := by
sorry

theorem difference_of_roots_x2_minus_7x_plus_9 :
  let x₁ := (7 + Real.sqrt 13) / 2
  let x₂ := (7 - Real.sqrt 13) / 2
  x₁^2 - 7*x₁ + 9 = 0 ∧ x₂^2 - 7*x₂ + 9 = 0 →
  |x₁ - x₂| = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_difference_of_roots_x2_minus_7x_plus_9_l3041_304154


namespace NUMINAMATH_CALUDE_parabola_c_value_l3041_304179

/-- Represents a parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 3 = 4 →  -- vertex at (4, 3)
  p.x_coord 5 = 2 →  -- passes through (2, 5)
  p.c = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3041_304179


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3041_304110

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 4}

-- Define set N
def N : Finset Nat := {1, 3, 5}

-- Theorem statement
theorem intersection_complement_equality :
  N ∩ (U \ M) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3041_304110


namespace NUMINAMATH_CALUDE_min_value_expression_l3041_304135

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) * (1 / x + 4 / y) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3041_304135


namespace NUMINAMATH_CALUDE_fraction_problem_l3041_304182

theorem fraction_problem (N : ℝ) (f : ℝ) : 
  N = 180 →
  6 + (1/2) * (1/3) * f * N = (1/15) * N →
  f = 1/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l3041_304182


namespace NUMINAMATH_CALUDE_percentage_reduction_price_increase_for_target_profit_price_increase_for_max_profit_maximum_daily_profit_l3041_304109

-- Define the original price and final price
def original_price : ℝ := 50
def final_price : ℝ := 32

-- Define the profit per kilogram and initial daily sales
def profit_per_kg : ℝ := 10
def initial_daily_sales : ℝ := 500

-- Define the reduction in sales per yuan increase
def sales_reduction_per_yuan : ℝ := 20

-- Define the target daily profit
def target_daily_profit : ℝ := 6000

-- Part 1: Percentage reduction
theorem percentage_reduction : 
  ∃ x : ℝ, x > 0 ∧ x < 1 ∧ original_price * (1 - x)^2 = final_price ∧ x = 0.2 := by sorry

-- Part 2: Price increase for target profit
theorem price_increase_for_target_profit :
  ∃ x : ℝ, x > 0 ∧ (profit_per_kg + x) * (initial_daily_sales - sales_reduction_per_yuan * x) = target_daily_profit ∧
  (∀ y : ℝ, y > 0 ∧ (profit_per_kg + y) * (initial_daily_sales - sales_reduction_per_yuan * y) = target_daily_profit → x ≤ y) ∧
  x = 5 := by sorry

-- Part 3: Price increase for maximum profit
def profit_function (x : ℝ) : ℝ := (profit_per_kg + x) * (initial_daily_sales - sales_reduction_per_yuan * x)

theorem price_increase_for_max_profit :
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, profit_function y ≤ profit_function x) ∧ x = 7.5 := by sorry

-- Part 4: Maximum daily profit
theorem maximum_daily_profit :
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, profit_function y ≤ profit_function x) ∧ profit_function x = 6125 := by sorry

end NUMINAMATH_CALUDE_percentage_reduction_price_increase_for_target_profit_price_increase_for_max_profit_maximum_daily_profit_l3041_304109


namespace NUMINAMATH_CALUDE_remainder_cd_mod_m_l3041_304107

theorem remainder_cd_mod_m (m c d : ℤ) : 
  m > 0 → 
  ∃ c_inv d_inv : ℤ, (c * c_inv ≡ 1 [ZMOD m]) ∧ (d * d_inv ≡ 1 [ZMOD m]) →
  d ≡ 2 * c_inv [ZMOD m] →
  c * d ≡ 2 [ZMOD m] := by
sorry

end NUMINAMATH_CALUDE_remainder_cd_mod_m_l3041_304107


namespace NUMINAMATH_CALUDE_soccer_team_boys_l3041_304139

/-- Proves the number of boys on a soccer team given certain conditions -/
theorem soccer_team_boys (total : ℕ) (attendees : ℕ) : 
  total = 30 → 
  attendees = 20 → 
  ∃ (boys girls : ℕ), 
    boys + girls = total ∧ 
    boys + (girls / 3) = attendees ∧
    boys = 15 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_boys_l3041_304139


namespace NUMINAMATH_CALUDE_pyramid_volume_is_2_root2_div_3_l3041_304114

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a pyramid with vertex P and base ABC -/
structure Pyramid where
  P : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Check if a triangle is equilateral -/
def isEquilateral (A B C : Point3D) : Prop := 
  distance A B = distance B C ∧ distance B C = distance C A

/-- Calculate the angle between three points -/
def angle (A P C : Point3D) : ℝ := sorry

/-- Calculate the volume of a pyramid -/
def pyramidVolume (p : Pyramid) : ℝ := sorry

theorem pyramid_volume_is_2_root2_div_3 (p : Pyramid) : 
  isEquilateral p.A p.B p.C →
  distance p.P p.A = distance p.P p.B ∧ 
  distance p.P p.B = distance p.P p.C →
  distance p.A p.B = 2 →
  angle p.A p.P p.C = Real.pi / 2 →
  pyramidVolume p = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_is_2_root2_div_3_l3041_304114


namespace NUMINAMATH_CALUDE_line_b_production_l3041_304112

/-- Represents a production line in a factory -/
inductive ProductionLine
| A
| B
| C

/-- Represents the production of a factory with three production lines -/
structure FactoryProduction where
  total : ℕ
  lines : ProductionLine → ℕ
  sum_eq_total : lines ProductionLine.A + lines ProductionLine.B + lines ProductionLine.C = total
  arithmetic_seq : ∃ d : ℤ, 
    (lines ProductionLine.B : ℤ) - (lines ProductionLine.A : ℤ) = d ∧
    (lines ProductionLine.C : ℤ) - (lines ProductionLine.B : ℤ) = d

/-- The theorem stating the production of line B given the conditions -/
theorem line_b_production (fp : FactoryProduction) 
  (h_total : fp.total = 16800) : 
  fp.lines ProductionLine.B = 5600 := by
  sorry

end NUMINAMATH_CALUDE_line_b_production_l3041_304112


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3041_304103

theorem largest_constant_inequality (C : ℝ) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z)) ↔ C ≤ Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3041_304103


namespace NUMINAMATH_CALUDE_largest_divisor_of_product_l3041_304165

theorem largest_divisor_of_product (n : ℕ) (h : Even n) (h2 : n > 0) :
  ∃ (k : ℕ), k ≤ 15 ∧ 
  (∀ (m : ℕ), m ≤ k → (m ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+11))) ∧
  (∀ (m : ℕ), m > 15 → ∃ (p : ℕ), Even p ∧ p > 0 ∧ ¬(m ∣ (p+1)*(p+3)*(p+5)*(p+7)*(p+11))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_product_l3041_304165


namespace NUMINAMATH_CALUDE_water_percentage_is_15_l3041_304146

/-- Calculates the percentage of water in a mixture of three liquids -/
def water_percentage_in_mixture (a_percentage : ℚ) (b_percentage : ℚ) (c_percentage : ℚ) 
  (a_parts : ℚ) (b_parts : ℚ) (c_parts : ℚ) : ℚ :=
  ((a_percentage * a_parts + b_percentage * b_parts + c_percentage * c_parts) / 
   (a_parts + b_parts + c_parts)) * 100

/-- Theorem stating that the percentage of water in the given mixture is 15% -/
theorem water_percentage_is_15 : 
  water_percentage_in_mixture (10/100) (15/100) (25/100) 4 3 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_water_percentage_is_15_l3041_304146


namespace NUMINAMATH_CALUDE_jimmy_father_emails_l3041_304125

/-- The number of emails Jimmy's father receives per day before subscribing to the news channel -/
def initial_emails_per_day : ℕ := 20

/-- The number of additional emails per day after subscribing to the news channel -/
def additional_emails_per_day : ℕ := 5

/-- The total number of days in April -/
def days_in_april : ℕ := 30

/-- The day in April when Jimmy's father subscribed to the news channel -/
def subscription_day : ℕ := days_in_april / 2

theorem jimmy_father_emails :
  (subscription_day * initial_emails_per_day) +
  ((days_in_april - subscription_day) * (initial_emails_per_day + additional_emails_per_day)) = 675 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_father_emails_l3041_304125


namespace NUMINAMATH_CALUDE_negative_roots_range_l3041_304194

theorem negative_roots_range (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ (1/2)^x = 3*a + 2) → a > -1/3 :=
by sorry

end NUMINAMATH_CALUDE_negative_roots_range_l3041_304194


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l3041_304142

/-- Given a circle C' with equation x^2 - 14x + y^2 + 16y + 100 = 0,
    prove that the sum of its center coordinates and radius is -1 + √13 -/
theorem circle_center_radius_sum :
  ∃ (a' b' r' : ℝ),
    (∀ (x y : ℝ), x^2 - 14*x + y^2 + 16*y + 100 = 0 ↔ (x - a')^2 + (y - b')^2 = r'^2) →
    a' + b' + r' = -1 + Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l3041_304142


namespace NUMINAMATH_CALUDE_circle_center_l3041_304176

def circle_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x + 4 * y^2 - 24 * y - 36 = 0

def is_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y → (x - h)^2 + (y - k)^2 = 1

theorem circle_center : is_center 1 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l3041_304176


namespace NUMINAMATH_CALUDE_chess_tournament_ties_l3041_304189

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  num_players : Nat
  points_win : Rat
  points_loss : Rat
  points_tie : Rat
  total_games : Nat
  total_points : Rat
  best_three_points : Rat
  last_nine_points : Rat

/-- The main theorem to be proved -/
theorem chess_tournament_ties (t : ChessTournament) : 
  t.num_players = 14 ∧ 
  t.points_win = 1 ∧ 
  t.points_loss = 0 ∧ 
  t.points_tie = 1/2 ∧ 
  t.total_games = 91 ∧ 
  t.total_points = 91 ∧
  t.best_three_points = t.last_nine_points ∧
  t.best_three_points = 36 →
  ∃ (num_ties : Nat), num_ties = 29 ∧ 
    (∀ (other_num_ties : Nat), other_num_ties > num_ties → 
      ¬(∃ (valid_tournament : ChessTournament), 
        valid_tournament.num_players = 14 ∧
        valid_tournament.points_win = 1 ∧
        valid_tournament.points_loss = 0 ∧
        valid_tournament.points_tie = 1/2 ∧
        valid_tournament.total_games = 91 ∧
        valid_tournament.total_points = 91 ∧
        valid_tournament.best_three_points = valid_tournament.last_nine_points ∧
        valid_tournament.best_three_points = 36)) :=
by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_ties_l3041_304189


namespace NUMINAMATH_CALUDE_range_of_f_range_of_a_l3041_304181

-- Define the functions
def f (x : ℝ) : ℝ := 2 * abs (x - 1) - abs (x - 4)

def g (x a : ℝ) : ℝ := 2 * abs (x - 1) - abs (x - a)

-- State the theorems
theorem range_of_f : Set.range f = Set.Ici (-3) := by sorry

theorem range_of_a (h : ∀ x : ℝ, g x a ≥ -1) : a ∈ Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_a_l3041_304181


namespace NUMINAMATH_CALUDE_distance_to_fountain_is_30_l3041_304191

/-- The distance from Mrs. Hilt's desk to the water fountain -/
def distance_to_fountain : ℕ := sorry

/-- The total distance Mrs. Hilt walks for all trips to the fountain -/
def total_distance : ℕ := 120

/-- The number of times Mrs. Hilt goes to the water fountain -/
def number_of_trips : ℕ := 4

/-- Theorem stating that the distance to the fountain is 30 feet -/
theorem distance_to_fountain_is_30 : 
  distance_to_fountain = total_distance / number_of_trips :=
sorry

end NUMINAMATH_CALUDE_distance_to_fountain_is_30_l3041_304191


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3041_304151

theorem algebraic_expression_value (a : ℝ) (h : a^2 + 2023*a - 1 = 0) :
  a*(a+1)*(a-1) + 2023*a^2 + 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3041_304151


namespace NUMINAMATH_CALUDE_boys_camp_problem_l3041_304130

theorem boys_camp_problem (total : ℕ) 
  (h1 : (total : ℚ) * (1/5) = (total : ℚ) * (20/100))
  (h2 : (total : ℚ) * (1/5) * (3/10) = (total : ℚ) * (1/5) * (30/100))
  (h3 : (total : ℚ) * (1/5) * (7/10) = 35) :
  total = 250 := by sorry

end NUMINAMATH_CALUDE_boys_camp_problem_l3041_304130


namespace NUMINAMATH_CALUDE_vector_simplification_l3041_304120

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem vector_simplification :
  (1 / 2 : ℝ) • ((2 : ℝ) • a + (8 : ℝ) • b) - ((4 : ℝ) • a - (2 : ℝ) • b) = (6 : ℝ) • b - (3 : ℝ) • a :=
by sorry

end NUMINAMATH_CALUDE_vector_simplification_l3041_304120


namespace NUMINAMATH_CALUDE_right_handed_players_count_l3041_304127

theorem right_handed_players_count (total_players throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 52)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) :
  throwers + ((total_players - throwers) * 2 / 3) = 64 := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l3041_304127


namespace NUMINAMATH_CALUDE_min_distance_to_parabola_l3041_304115

/-- Rectilinear distance between two points -/
def rectilinear_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

/-- A point on the parabola x² = y -/
def parabola_point (t : ℝ) : ℝ × ℝ := (t, t^2)

/-- The fixed point M(-1, 0) -/
def M : ℝ × ℝ := (-1, 0)

/-- Theorem: The minimum rectilinear distance from M(-1, 0) to the parabola x² = y is 3/4 -/
theorem min_distance_to_parabola :
  ∀ t : ℝ, rectilinear_distance (M.1) (M.2) (parabola_point t).1 (parabola_point t).2 ≥ 3/4 ∧
  ∃ t₀ : ℝ, rectilinear_distance (M.1) (M.2) (parabola_point t₀).1 (parabola_point t₀).2 = 3/4 :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_parabola_l3041_304115


namespace NUMINAMATH_CALUDE_base3_12012_equals_140_l3041_304134

def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

theorem base3_12012_equals_140 :
  base3ToBase10 [2, 1, 0, 2, 1] = 140 := by
  sorry

end NUMINAMATH_CALUDE_base3_12012_equals_140_l3041_304134


namespace NUMINAMATH_CALUDE_distance_on_quadratic_curve_l3041_304121

/-- The distance between two points on a quadratic curve -/
theorem distance_on_quadratic_curve (m k a b c d : ℝ) :
  b = m * a^2 + k →
  d = m * c^2 + k →
  Real.sqrt ((c - a)^2 + (d - b)^2) = |c - a| * Real.sqrt (1 + m^2 * (c + a)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_on_quadratic_curve_l3041_304121


namespace NUMINAMATH_CALUDE_S_is_infinite_l3041_304152

-- Define the set of points that satisfy the conditions
def S : Set (ℚ × ℚ) := {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 ≤ 5}

-- Theorem: The set S is infinite
theorem S_is_infinite : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_S_is_infinite_l3041_304152


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3041_304108

def U : Set Nat := {1, 2, 3, 4, 5}
def P : Set Nat := {2, 4}
def Q : Set Nat := {1, 3, 4, 6}

theorem complement_intersection_theorem :
  (U \ P) ∩ Q = {1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3041_304108


namespace NUMINAMATH_CALUDE_factorial_expression_equals_2015_l3041_304122

-- Define factorial function
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_expression_equals_2015 : 
  (factorial (factorial 2014 - 1) * factorial 2015) / factorial (factorial 2014) = 2015 := by
  sorry

end NUMINAMATH_CALUDE_factorial_expression_equals_2015_l3041_304122


namespace NUMINAMATH_CALUDE_exactly_one_double_root_l3041_304186

/-- The function f(x) representing the left side of the equation -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + 2)^2 * (x + 7)^2 + a

/-- The theorem stating the condition for exactly one double-root -/
theorem exactly_one_double_root (a : ℝ) : 
  (∃! x : ℝ, f a x = 0 ∧ (∀ y : ℝ, y ≠ x → f a y > 0)) ↔ a = -39.0625 := by sorry

end NUMINAMATH_CALUDE_exactly_one_double_root_l3041_304186


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3041_304138

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3 * x + y = 18) : 
  x^2 - y^2 = -72 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3041_304138


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l3041_304178

theorem unique_x_with_three_prime_factors (x n : ℕ) : 
  x = 5^n - 1 ∧ 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ 
    x = 2^(Nat.log 2 x) * 11 * p * q) →
  x = 3124 :=
by sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l3041_304178


namespace NUMINAMATH_CALUDE_sandy_molly_age_ratio_l3041_304117

/-- The ratio of ages between two people -/
def age_ratio (age1 : ℕ) (age2 : ℕ) : ℚ := age1 / age2

/-- Sandy's age -/
def sandy_age : ℕ := 49

/-- Age difference between Molly and Sandy -/
def age_difference : ℕ := 14

/-- Molly's age -/
def molly_age : ℕ := sandy_age + age_difference

theorem sandy_molly_age_ratio :
  age_ratio sandy_age molly_age = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sandy_molly_age_ratio_l3041_304117
