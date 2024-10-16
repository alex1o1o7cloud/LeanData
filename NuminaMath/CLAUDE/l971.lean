import Mathlib

namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l971_97163

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - 4 * x + m = 0 ∧ 
   ∀ y : ℝ, 2 * y^2 - 4 * y + m = 0 → y = x) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l971_97163


namespace NUMINAMATH_CALUDE_smallest_set_size_existence_of_set_smallest_set_size_is_eight_l971_97135

theorem smallest_set_size (n : ℕ) (h : n ≥ 5) :
  (∃ (S : Finset (ℕ × ℕ)),
    S.card = n ∧
    (∀ (p : ℕ × ℕ), p ∈ S → p.1 ∈ Finset.range 4 ∧ p.2 ∈ Finset.range 4) ∧
    (∀ (p q : ℕ × ℕ), p ∈ S → q ∈ S →
      ∃ (r : ℕ × ℕ), r ∈ S ∧
        4 ∣ (p.1 + q.1 - r.1) ∧
        4 ∣ (p.2 + q.2 - r.2))) →
  n ≥ 8 := by sorry

theorem existence_of_set :
  ∃ (S : Finset (ℕ × ℕ)),
    S.card = 8 ∧
    (∀ (p : ℕ × ℕ), p ∈ S → p.1 ∈ Finset.range 4 ∧ p.2 ∈ Finset.range 4) ∧
    (∀ (p q : ℕ × ℕ), p ∈ S → q ∈ S →
      ∃ (r : ℕ × ℕ), r ∈ S ∧
        4 ∣ (p.1 + q.1 - r.1) ∧
        4 ∣ (p.2 + q.2 - r.2)) := by sorry

theorem smallest_set_size_is_eight :
  (∃ (n : ℕ), n ≥ 5 ∧
    (∃ (S : Finset (ℕ × ℕ)),
      S.card = n ∧
      (∀ (p : ℕ × ℕ), p ∈ S → p.1 ∈ Finset.range 4 ∧ p.2 ∈ Finset.range 4) ∧
      (∀ (p q : ℕ × ℕ), p ∈ S → q ∈ S →
        ∃ (r : ℕ × ℕ), r ∈ S ∧
          4 ∣ (p.1 + q.1 - r.1) ∧
          4 ∣ (p.2 + q.2 - r.2)))) →
  (∀ (m : ℕ), m ≥ 5 →
    (∃ (S : Finset (ℕ × ℕ)),
      S.card = m ∧
      (∀ (p : ℕ × ℕ), p ∈ S → p.1 ∈ Finset.range 4 ∧ p.2 ∈ Finset.range 4) ∧
      (∀ (p q : ℕ × ℕ), p ∈ S → q ∈ S →
        ∃ (r : ℕ × ℕ), r ∈ S ∧
          4 ∣ (p.1 + q.1 - r.1) ∧
          4 ∣ (p.2 + q.2 - r.2))) →
    m ≥ 8) ∧
  (∃ (S : Finset (ℕ × ℕ)),
    S.card = 8 ∧
    (∀ (p : ℕ × ℕ), p ∈ S → p.1 ∈ Finset.range 4 ∧ p.2 ∈ Finset.range 4) ∧
    (∀ (p q : ℕ × ℕ), p ∈ S → q ∈ S →
      ∃ (r : ℕ × ℕ), r ∈ S ∧
        4 ∣ (p.1 + q.1 - r.1) ∧
        4 ∣ (p.2 + q.2 - r.2))) := by sorry

end NUMINAMATH_CALUDE_smallest_set_size_existence_of_set_smallest_set_size_is_eight_l971_97135


namespace NUMINAMATH_CALUDE_fifteen_power_equals_R_S_power_l971_97121

theorem fifteen_power_equals_R_S_power (a b : ℤ) (R S : ℝ) 
  (hR : R = 3^a) (hS : S = 5^b) : 15^(a*b) = R^b * S^a := by
  sorry

end NUMINAMATH_CALUDE_fifteen_power_equals_R_S_power_l971_97121


namespace NUMINAMATH_CALUDE_odd_function_properties_l971_97144

-- Define the function h
noncomputable def h : ℝ → ℝ := fun x ↦ 2^x

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x ↦ (1 - h x) / (1 + h x)

-- State the theorem
theorem odd_function_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (h 2 = 4) ∧             -- h(2) = 4
  (∀ x, f x = (1 - 2^x) / (1 + 2^x)) ∧  -- Analytical form of f
  (∀ x, f (2*x - 1) > f (x + 1) ↔ x < 2/3) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_properties_l971_97144


namespace NUMINAMATH_CALUDE_water_displacement_theorem_l971_97145

/-- Represents a cylindrical barrel --/
structure Barrel where
  radius : ℝ
  height : ℝ

/-- Represents a cube --/
structure Cube where
  side_length : ℝ

/-- Calculates the volume of water displaced by a cube in a barrel --/
def water_displaced (barrel : Barrel) (cube : Cube) : ℝ :=
  cube.side_length ^ 3

/-- The main theorem about water displacement and its square --/
theorem water_displacement_theorem (barrel : Barrel) (cube : Cube)
    (h1 : barrel.radius = 5)
    (h2 : barrel.height = 15)
    (h3 : cube.side_length = 10) :
    let v := water_displaced barrel cube
    v = 1000 ∧ v^2 = 1000000 := by
  sorry

#check water_displacement_theorem

end NUMINAMATH_CALUDE_water_displacement_theorem_l971_97145


namespace NUMINAMATH_CALUDE_tiffany_miles_per_day_l971_97198

theorem tiffany_miles_per_day (T : ℚ) : 
  (7 : ℚ) = 3 * T + 3 * (1/3 : ℚ) + 0 → T = 2 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_miles_per_day_l971_97198


namespace NUMINAMATH_CALUDE_base_5_to_binary_44_l971_97119

def base_5_to_decimal (n : ℕ) : ℕ := 
  (n / 10) * 5 + (n % 10)

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem base_5_to_binary_44 :
  decimal_to_binary (base_5_to_decimal 44) = [1, 1, 0, 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_base_5_to_binary_44_l971_97119


namespace NUMINAMATH_CALUDE_range_of_a_l971_97162

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 5 → x^2 - 2*x + a ≥ 0) ↔ a ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l971_97162


namespace NUMINAMATH_CALUDE_parabola_triangle_circumradius_range_l971_97195

/-- A point on a parabola y = x^2 -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y = x^2

/-- Triangle on a parabola y = x^2 -/
structure ParabolaTriangle where
  A : ParabolaPoint
  B : ParabolaPoint
  C : ParabolaPoint
  distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C

/-- The circumradius of a triangle -/
def circumradius (t : ParabolaTriangle) : ℝ :=
  sorry  -- Definition of circumradius

theorem parabola_triangle_circumradius_range :
  ∀ (t : ParabolaTriangle), circumradius t > 1/2 ∧
  ∀ (r : ℝ), r > 1/2 → ∃ (t : ParabolaTriangle), circumradius t = r :=
by sorry

end NUMINAMATH_CALUDE_parabola_triangle_circumradius_range_l971_97195


namespace NUMINAMATH_CALUDE_f_properties_l971_97186

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

theorem f_properties :
  (∃ (x : ℝ), -2 < x ∧ x < 2 ∧ f x = 5) ∧
  (∀ (y : ℝ), -2 < y ∧ y < 2 → f y ≤ 5) ∧
  (¬ ∃ (z : ℝ), -2 < z ∧ z < 2 ∧ ∀ (w : ℝ), -2 < w ∧ w < 2 → f z ≤ f w) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l971_97186


namespace NUMINAMATH_CALUDE_quiz_competition_outcomes_quiz_competition_proof_l971_97151

def participants : Nat := 6

theorem quiz_competition_outcomes (rita_not_third : Bool) : Nat :=
  participants * (participants - 1) * (participants - 2)

theorem quiz_competition_proof :
  quiz_competition_outcomes true = 120 := by sorry

end NUMINAMATH_CALUDE_quiz_competition_outcomes_quiz_competition_proof_l971_97151


namespace NUMINAMATH_CALUDE_chess_match_probability_l971_97129

theorem chess_match_probability (p_win p_draw : ℝ) 
  (h1 : p_win = 0.4) 
  (h2 : p_draw = 0.2) : 
  p_win + p_draw = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_chess_match_probability_l971_97129


namespace NUMINAMATH_CALUDE_percentage_in_quarters_l971_97153

def dimes : ℕ := 80
def quarters : ℕ := 30
def nickels : ℕ := 40

def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5

def total_value : ℕ := dimes * dime_value + quarters * quarter_value + nickels * nickel_value
def quarters_value : ℕ := quarters * quarter_value

theorem percentage_in_quarters : 
  (quarters_value : ℚ) / total_value * 100 = 3/7 * 100 := by sorry

end NUMINAMATH_CALUDE_percentage_in_quarters_l971_97153


namespace NUMINAMATH_CALUDE_exactly_three_non_congruent_triangles_l971_97114

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if two triangles are congruent -/
def congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all triangles with perimeter 11 -/
def triangles_with_perimeter_11 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 11}

/-- The theorem to be proved -/
theorem exactly_three_non_congruent_triangles :
  ∃ (t1 t2 t3 : IntTriangle),
    t1 ∈ triangles_with_perimeter_11 ∧
    t2 ∈ triangles_with_perimeter_11 ∧
    t3 ∈ triangles_with_perimeter_11 ∧
    ¬congruent t1 t2 ∧ ¬congruent t1 t3 ∧ ¬congruent t2 t3 ∧
    ∀ (t : IntTriangle),
      t ∈ triangles_with_perimeter_11 →
      (congruent t t1 ∨ congruent t t2 ∨ congruent t t3) :=
by sorry

end NUMINAMATH_CALUDE_exactly_three_non_congruent_triangles_l971_97114


namespace NUMINAMATH_CALUDE_infinitely_many_square_averages_l971_97196

theorem infinitely_many_square_averages :
  ∃ f : ℕ → ℕ, 
    (f 0 = 1) ∧ 
    (∀ k : ℕ, f k < f (k + 1)) ∧
    (∀ k : ℕ, ∃ m : ℕ, (f k * (f k + 1) * (2 * f k + 1)) / 6 = m^2 * f k) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_square_averages_l971_97196


namespace NUMINAMATH_CALUDE_rectangle_count_l971_97138

/-- The number of different rectangles in a 5x5 grid -/
def num_rectangles : ℕ := 100

/-- The number of rows in the grid -/
def num_rows : ℕ := 5

/-- The number of columns in the grid -/
def num_columns : ℕ := 5

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem rectangle_count :
  num_rectangles = choose_two num_rows * choose_two num_columns :=
sorry

end NUMINAMATH_CALUDE_rectangle_count_l971_97138


namespace NUMINAMATH_CALUDE_systematic_sampling_l971_97179

theorem systematic_sampling (total_students : ℕ) (sample_size : ℕ) (interval : ℕ) (start : ℕ) :
  total_students = 800 →
  sample_size = 50 →
  interval = 16 →
  start = 7 →
  ∃ (n : ℕ), n ≤ 4 ∧ 
    (start + (n - 1) * interval ≥ 49) ∧ 
    (start + (n - 1) * interval ≤ 64) ∧
    (start + (n - 1) * interval = 55) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l971_97179


namespace NUMINAMATH_CALUDE_triangle_side_length_l971_97148

theorem triangle_side_length (a b c : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) →  -- positive side lengths
  (|a - 7| + (b - 2)^2 = 0) →    -- given equation
  (∃ n : ℕ, c = 2*n + 1) →      -- c is odd
  (a + b > c) →                 -- triangle inequality
  (a + c > b) →                 -- triangle inequality
  (b + c > a) →                 -- triangle inequality
  (c = 7) :=                    -- conclusion
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l971_97148


namespace NUMINAMATH_CALUDE_number_puzzle_l971_97133

theorem number_puzzle : ∃ x : ℚ, 45 + 3 * x = 60 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l971_97133


namespace NUMINAMATH_CALUDE_original_bales_count_l971_97132

/-- The number of bales Jason stacked today -/
def bales_stacked : ℕ := 23

/-- The total number of bales in the barn after Jason stacked -/
def total_bales : ℕ := 96

/-- The original number of bales in the barn -/
def original_bales : ℕ := total_bales - bales_stacked

theorem original_bales_count : original_bales = 73 := by
  sorry

end NUMINAMATH_CALUDE_original_bales_count_l971_97132


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l971_97192

/-- Two 2D vectors are parallel if their corresponding components are proportional -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (-1, 1)
  let b : ℝ × ℝ := (3, m)
  parallel a b → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l971_97192


namespace NUMINAMATH_CALUDE_rhino_horn_segment_area_l971_97100

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents the "rhino's horn segment" region -/
structure RhinoHornSegment where
  largeCircle : Circle
  smallCircle : Circle
  basePoint : Point
  endPoint : Point

/-- Calculates the area of the "rhino's horn segment" -/
def rhinoHornSegmentArea (r : RhinoHornSegment) : ℝ :=
  sorry

/-- The main theorem stating that the area of the "rhino's horn segment" is 2π -/
theorem rhino_horn_segment_area :
  let r := RhinoHornSegment.mk
    (Circle.mk (Point.mk 0 0) 4)
    (Circle.mk (Point.mk 0 2) 2)
    (Point.mk 0 0)
    (Point.mk 4 0)
  rhinoHornSegmentArea r = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_rhino_horn_segment_area_l971_97100


namespace NUMINAMATH_CALUDE_condition_for_a_greater_than_b_l971_97101

-- Define the property of being sufficient but not necessary
def sufficient_but_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

-- Theorem statement
theorem condition_for_a_greater_than_b (a b : ℝ) :
  sufficient_but_not_necessary (a > b + 1) (a > b) := by
  sorry

end NUMINAMATH_CALUDE_condition_for_a_greater_than_b_l971_97101


namespace NUMINAMATH_CALUDE_water_tank_capacity_l971_97139

/-- The total capacity of a water tank in gallons -/
def tank_capacity : ℝ → Prop := λ T =>
  -- When the tank is 40% full, it contains 36 gallons less than when it is 70% full
  0.7 * T - 0.4 * T = 36

theorem water_tank_capacity : ∃ T : ℝ, tank_capacity T ∧ T = 120 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l971_97139


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l971_97157

theorem cubic_sum_theorem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) : 
  a^3 + b^3 + c^3 = -36 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l971_97157


namespace NUMINAMATH_CALUDE_determinant_equality_l971_97193

theorem determinant_equality (p q r s : ℝ) : 
  p * s - q * r = 10 → (p + 2*r) * s - (q + 2*s) * r = 10 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l971_97193


namespace NUMINAMATH_CALUDE_married_men_fraction_l971_97174

theorem married_men_fraction (total_women : ℕ) (h_total_women_pos : total_women > 0) :
  let single_women : ℕ := (3 * total_women) / 7
  let married_women : ℕ := total_women - single_women
  let married_men : ℕ := married_women
  let total_people : ℕ := total_women + married_men
  (↑single_women : ℚ) / total_women = 3 / 7 →
  (↑married_men : ℚ) / total_people = 4 / 11 :=
by sorry

end NUMINAMATH_CALUDE_married_men_fraction_l971_97174


namespace NUMINAMATH_CALUDE_kaleb_initial_tickets_l971_97181

def ticket_cost : ℕ := 9
def ferris_wheel_cost : ℕ := 27
def remaining_tickets : ℕ := 3

def initial_tickets : ℕ := ferris_wheel_cost / ticket_cost + remaining_tickets

theorem kaleb_initial_tickets : initial_tickets = 6 := by sorry

end NUMINAMATH_CALUDE_kaleb_initial_tickets_l971_97181


namespace NUMINAMATH_CALUDE_other_divisor_problem_l971_97169

theorem other_divisor_problem (n : ℕ) (h1 : n = 174) : 
  ∃ (x : ℕ), x ≠ 5 ∧ x < 170 ∧ 
  (∀ y : ℕ, y < 170 → y ≠ 5 → n % y = 4 → y ≤ x) ∧
  n % x = 4 ∧ n % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_other_divisor_problem_l971_97169


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l971_97158

theorem quadratic_function_minimum (a b c : ℝ) 
  (h1 : b > a) 
  (h2 : a > 0) 
  (h3 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) : 
  (a + b + c) / (b - a) ≥ 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l971_97158


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l971_97150

theorem complex_number_in_fourth_quadrant (a : ℝ) :
  let z : ℂ := 3 - Complex.I * (a^2 + 1)
  z.re > 0 ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l971_97150


namespace NUMINAMATH_CALUDE_marbles_ratio_l971_97117

def marbles_problem (initial_marbles : ℕ) (current_marbles : ℕ) (brother_marbles : ℕ) : Prop :=
  let savanna_marbles := 3 * current_marbles
  let sister_marbles := initial_marbles - current_marbles - brother_marbles - savanna_marbles
  (sister_marbles : ℚ) / brother_marbles = 2

theorem marbles_ratio :
  marbles_problem 300 30 60 := by
  sorry

end NUMINAMATH_CALUDE_marbles_ratio_l971_97117


namespace NUMINAMATH_CALUDE_pizza_cost_l971_97126

theorem pizza_cost (initial_amount : ℕ) (return_amount : ℕ) (juice_cost : ℕ) (juice_quantity : ℕ) (pizza_quantity : ℕ) :
  initial_amount = 50 ∧
  return_amount = 22 ∧
  juice_cost = 2 ∧
  juice_quantity = 2 ∧
  pizza_quantity = 2 →
  (initial_amount - return_amount - juice_cost * juice_quantity) / pizza_quantity = 12 :=
by sorry

end NUMINAMATH_CALUDE_pizza_cost_l971_97126


namespace NUMINAMATH_CALUDE_mark_reading_time_l971_97154

/-- Calculates Mark's total weekly reading time given his daily reading time and weekly increase. -/
def weekly_reading_time (x : ℝ) (y : ℝ) : ℝ :=
  7 * x + y

/-- Theorem stating that Mark's total weekly reading time is 7x + y hours -/
theorem mark_reading_time (x y : ℝ) :
  weekly_reading_time x y = 7 * x + y := by
  sorry

end NUMINAMATH_CALUDE_mark_reading_time_l971_97154


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l971_97178

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (surface_area : 2 * (a * b + b * c + a * c) = 54)
  (edge_length : 4 * (a + b + c) = 40) :
  ∃ d : ℝ, d^2 = a^2 + b^2 + c^2 ∧ d = Real.sqrt 46 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l971_97178


namespace NUMINAMATH_CALUDE_house_number_painting_cost_l971_97164

/-- Represents a side of the street with houses -/
structure StreetSide where
  start : ℕ
  diff : ℕ
  count : ℕ

/-- Calculates the cost of painting numbers for a given street side -/
def paintCost (side : StreetSide) : ℕ := sorry

/-- The problem statement -/
theorem house_number_painting_cost :
  let southSide : StreetSide := { start := 5, diff := 7, count := 25 }
  let northSide : StreetSide := { start := 2, diff := 8, count := 25 }
  paintCost southSide + paintCost northSide = 123 := by sorry

end NUMINAMATH_CALUDE_house_number_painting_cost_l971_97164


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l971_97111

def sequence_a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * sequence_a (n + 1) + sequence_a n

theorem divisibility_equivalence (k n : ℕ) :
  (2^k : ℤ) ∣ sequence_a n ↔ 2^k ∣ n := by sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l971_97111


namespace NUMINAMATH_CALUDE_horner_method_f_2_l971_97160

def f (x : ℝ) : ℝ := 2*x^5 - 3*x^3 + 2*x^2 + x - 3

def horner_v3 (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v0 := 2
  let v1 := v0 * x
  let v2 := (v1 - 3) * x + 2
  (v2 * x + 1) * x - 3

theorem horner_method_f_2 :
  horner_v3 f 2 = 12 := by sorry

end NUMINAMATH_CALUDE_horner_method_f_2_l971_97160


namespace NUMINAMATH_CALUDE_solution_is_eight_l971_97127

/-- The function f(x) = 2x - b -/
def f (x : ℝ) (b : ℝ) : ℝ := 2 * x - b

/-- The theorem stating that x = 8 is the solution to the equation -/
theorem solution_is_eight :
  ∃! x : ℝ, 2 * (f x 3) - 21 = f (x - 4) 3 :=
sorry

end NUMINAMATH_CALUDE_solution_is_eight_l971_97127


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l971_97156

theorem complex_subtraction_simplification :
  (4 - 3 * Complex.I) - (7 - 5 * Complex.I) = -3 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l971_97156


namespace NUMINAMATH_CALUDE_odd_prime_and_odd_natural_not_divide_l971_97166

theorem odd_prime_and_odd_natural_not_divide (p n : ℕ) : 
  Nat.Prime p → Odd p → Odd n → ¬(p * n + 1 ∣ p^p - 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_and_odd_natural_not_divide_l971_97166


namespace NUMINAMATH_CALUDE_opposite_of_four_l971_97102

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- Theorem statement
theorem opposite_of_four : opposite 4 = -4 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_four_l971_97102


namespace NUMINAMATH_CALUDE_local_science_students_percentage_l971_97136

/-- Proves that the percentage of local science students is 25% given the conditions of the problem -/
theorem local_science_students_percentage 
  (total_arts : ℕ) 
  (total_science : ℕ) 
  (total_commerce : ℕ) 
  (local_arts_percentage : ℚ) 
  (local_commerce_percentage : ℚ) 
  (total_local_percentage : ℚ) 
  (h1 : total_arts = 400) 
  (h2 : total_science = 100) 
  (h3 : total_commerce = 120) 
  (h4 : local_arts_percentage = 1/2) 
  (h5 : local_commerce_percentage = 17/20) 
  (h6 : total_local_percentage = 327/100) : 
  ∃ (local_science_percentage : ℚ), 
    local_science_percentage = 1/4 ∧ 
    (local_arts_percentage * total_arts + local_science_percentage * total_science + local_commerce_percentage * total_commerce) / (total_arts + total_science + total_commerce) = total_local_percentage := by
  sorry


end NUMINAMATH_CALUDE_local_science_students_percentage_l971_97136


namespace NUMINAMATH_CALUDE_min_sum_dimensions_l971_97120

theorem min_sum_dimensions (l w h : ℕ+) : 
  l * w * h = 3003 → l + w + h ≥ 45 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_l971_97120


namespace NUMINAMATH_CALUDE_row_swap_matrix_exists_l971_97183

open Matrix

theorem row_swap_matrix_exists : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
  N * A = ![![A 1 0, A 1 1], ![A 0 0, A 0 1]] := by
  sorry

end NUMINAMATH_CALUDE_row_swap_matrix_exists_l971_97183


namespace NUMINAMATH_CALUDE_square_difference_l971_97106

theorem square_difference : (39 : ℤ)^2 = 40^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l971_97106


namespace NUMINAMATH_CALUDE_podcast_ratio_l971_97167

def total_drive_time : ℕ := 360 -- in minutes
def first_podcast : ℕ := 45 -- in minutes
def third_podcast : ℕ := 105 -- in minutes
def fourth_podcast : ℕ := 60 -- in minutes
def next_podcast : ℕ := 60 -- in minutes

theorem podcast_ratio : 
  let second_podcast := total_drive_time - (first_podcast + third_podcast + fourth_podcast + next_podcast)
  (second_podcast : ℚ) / first_podcast = 2 := by
sorry

end NUMINAMATH_CALUDE_podcast_ratio_l971_97167


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l971_97108

theorem root_sum_reciprocal (α β γ : ℂ) : 
  (α^3 - 2*α^2 - α + 2 = 0) → 
  (β^3 - 2*β^2 - β + 2 = 0) → 
  (γ^3 - 2*γ^2 - γ + 2 = 0) → 
  (1 / (α + 2) + 1 / (β + 2) + 1 / (γ + 2) = -19 / 14) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l971_97108


namespace NUMINAMATH_CALUDE_melinda_original_cost_l971_97107

/-- Represents the original cost of clothing items before tax and discounts -/
def original_cost (jeans_price shirt_price jacket_price : ℝ) : ℝ :=
  jeans_price + shirt_price + jacket_price

/-- The theorem stating the original cost of Melinda's purchase -/
theorem melinda_original_cost :
  original_cost 14.50 9.50 21.00 = 45.00 := by
  sorry

end NUMINAMATH_CALUDE_melinda_original_cost_l971_97107


namespace NUMINAMATH_CALUDE_investment_problem_l971_97188

/-- Prove the existence and uniqueness of the investment amount and interest rate -/
theorem investment_problem :
  ∃! (P y : ℝ), P > 0 ∧ y > 0 ∧
    P * y * 2 / 100 = 800 ∧
    P * ((1 + y / 100)^2 - 1) = 820 ∧
    P = 8000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l971_97188


namespace NUMINAMATH_CALUDE_vector_magnitude_l971_97155

/-- Given two vectors a and b in ℝ² with an angle of π/3 between them,
    where a = (1, √3) and |a - 2b| = 2√3, prove that |b| = 2 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  (a.1 = 1 ∧ a.2 = Real.sqrt 3) →  -- a = (1, √3)
  (a.1 * b.1 + a.2 * b.2 = Real.sqrt 3 * Real.sqrt ((a.1^2 + a.2^2) * (b.1^2 + b.2^2))) →  -- angle between a and b is π/3
  ((a.1 - 2*b.1)^2 + (a.2 - 2*b.2)^2 = 12) →  -- |a - 2b| = 2√3
  Real.sqrt (b.1^2 + b.2^2) = 2  -- |b| = 2
:= by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l971_97155


namespace NUMINAMATH_CALUDE_constant_term_expansion_l971_97116

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function for the general term of the expansion
def generalTerm (r : ℕ) : ℤ :=
  (-1)^r * 2^(4 - r) * binomial 4 r

-- Theorem statement
theorem constant_term_expansion :
  generalTerm 2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l971_97116


namespace NUMINAMATH_CALUDE_isosceles_if_neg_one_root_side_c_value_l971_97130

/-- Triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation b(x²-1) + 2ax + c(x²+1) = 0 -/
def equation (t : Triangle) (x : ℝ) : Prop :=
  t.b * (x^2 - 1) + 2 * t.a * x + t.c * (x^2 + 1) = 0

theorem isosceles_if_neg_one_root (t : Triangle) :
  equation t (-1) → t.a = t.c :=
sorry

theorem side_c_value (t : Triangle) :
  (∃ x : ℝ, ∀ y : ℝ, equation t x ↔ y = x) →
  t.a = 5 →
  t.b = 12 →
  t.c = 13 :=
sorry

end NUMINAMATH_CALUDE_isosceles_if_neg_one_root_side_c_value_l971_97130


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_18_l971_97131

theorem least_five_digit_congruent_to_7_mod_18 :
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ n % 18 = 7 → n ≥ 10015 :=
by
  sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_18_l971_97131


namespace NUMINAMATH_CALUDE_raffle_winnings_l971_97161

theorem raffle_winnings (W : ℝ) (h1 : W > 0) (h2 : W / 2 - 2 + 114 = W) : 
  W - W / 2 - 2 = 110 := by
  sorry

end NUMINAMATH_CALUDE_raffle_winnings_l971_97161


namespace NUMINAMATH_CALUDE_rectangle_area_reduction_l971_97137

theorem rectangle_area_reduction (l w : ℝ) : 
  l > 0 ∧ w > 0 ∧ (l - 1) * w = 24 → l * (w - 1) = 18 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_reduction_l971_97137


namespace NUMINAMATH_CALUDE_replaced_person_weight_l971_97143

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (average_increase : ℝ) (weight_of_new_person : ℝ) : ℝ :=
  weight_of_new_person - 2 * average_increase

/-- Theorem stating that the weight of the replaced person is 65 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 4.5 74 = 65 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l971_97143


namespace NUMINAMATH_CALUDE_greatest_valid_sequence_length_l971_97185

/-- A sequence of distinct positive integers satisfying the given condition -/
def ValidSequence (s : Nat → Nat) (n : Nat) : Prop :=
  (∀ i j, i < n → j < n → i ≠ j → s i ≠ s j) ∧ 
  (∀ i, i < n - 1 → (s i) ^ (s (i + 1)) = (s (i + 1)) ^ (s (i + 2)))

/-- The theorem stating that 5 is the greatest positive integer satisfying the condition -/
theorem greatest_valid_sequence_length : 
  (∃ s : Nat → Nat, ValidSequence s 5) ∧ 
  (∀ n : Nat, n > 5 → ¬∃ s : Nat → Nat, ValidSequence s n) :=
sorry

end NUMINAMATH_CALUDE_greatest_valid_sequence_length_l971_97185


namespace NUMINAMATH_CALUDE_complex_addition_l971_97149

theorem complex_addition : (1 : ℂ) + 3*I + (2 : ℂ) - 4*I = 3 - I := by sorry

end NUMINAMATH_CALUDE_complex_addition_l971_97149


namespace NUMINAMATH_CALUDE_number_of_appliances_l971_97191

/-- Proves that the number of appliances in a batch is 34, given the purchase price,
    selling price, and total profit. -/
theorem number_of_appliances (purchase_price selling_price total_profit : ℕ) : 
  purchase_price = 230 →
  selling_price = 250 →
  total_profit = 680 →
  (total_profit / (selling_price - purchase_price) : ℕ) = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_appliances_l971_97191


namespace NUMINAMATH_CALUDE_geometric_mean_of_4_and_16_l971_97140

theorem geometric_mean_of_4_and_16 (x : ℝ) :
  x^2 = 4 * 16 → x = 8 ∨ x = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_4_and_16_l971_97140


namespace NUMINAMATH_CALUDE_distance_from_blast_site_l971_97134

/-- Proves the distance a man is from a blast site when he hears a second blast -/
theorem distance_from_blast_site (speed_of_sound : ℝ) (time_between_blasts : ℝ) (time_heard_second_blast : ℝ) : 
  speed_of_sound = 330 →
  time_between_blasts = 30 →
  time_heard_second_blast = 30 + 12 / 60 →
  speed_of_sound * (time_heard_second_blast - time_between_blasts) = 3960 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_blast_site_l971_97134


namespace NUMINAMATH_CALUDE_polynomial_expansion_l971_97122

theorem polynomial_expansion :
  ∀ x : ℝ, (4 * x^3 - 3 * x^2 + 2 * x + 7) * (5 * x^4 + x^3 - 3 * x + 9) =
    20 * x^7 - 27 * x^5 + 8 * x^4 + 45 * x^3 - 4 * x^2 + 51 * x + 196 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l971_97122


namespace NUMINAMATH_CALUDE_chessboard_separating_edges_l971_97152

/-- Represents a square on the chessboard -/
inductive Square
| White
| Black

/-- Represents the chessboard -/
def Chessboard (n : ℕ) := Fin n → Fin n → Square

/-- Counts the number of white squares on the border of the chessboard -/
def countWhiteBorderSquares (n : ℕ) (board : Chessboard n) : ℕ := sorry

/-- Counts the number of black squares on the border of the chessboard -/
def countBlackBorderSquares (n : ℕ) (board : Chessboard n) : ℕ := sorry

/-- Counts the number of edges inside the board that separate squares of different colors -/
def countSeparatingEdges (n : ℕ) (board : Chessboard n) : ℕ := sorry

/-- Main theorem: If a chessboard has at least n white and n black squares on its border,
    then there are at least n edges inside the board separating different colors -/
theorem chessboard_separating_edges (n : ℕ) (board : Chessboard n) :
  countWhiteBorderSquares n board ≥ n →
  countBlackBorderSquares n board ≥ n →
  countSeparatingEdges n board ≥ n := by sorry

end NUMINAMATH_CALUDE_chessboard_separating_edges_l971_97152


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l971_97159

theorem consecutive_integers_product_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 2720 → n + (n + 1) = 103 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l971_97159


namespace NUMINAMATH_CALUDE_f_bound_l971_97170

def f (x : ℝ) : ℝ := x^2 - x + 13

theorem f_bound (a x : ℝ) (h : |x - a| < 1) : |f x - f a| < 2 * (|a| + 1) := by
  sorry

end NUMINAMATH_CALUDE_f_bound_l971_97170


namespace NUMINAMATH_CALUDE_cube_root_simplification_l971_97146

theorem cube_root_simplification :
  (20^3 + 30^3 + 40^3 + 60^3 : ℝ)^(1/3) = 10 * 315^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l971_97146


namespace NUMINAMATH_CALUDE_min_year_exceed_300k_l971_97125

/-- Represents the linear regression equation for online shoppers --/
def online_shoppers (x : ℤ) : ℝ := 42 * x - 26

/-- Theorem: The minimum integer value of x for which the number of online shoppers exceeds 300 thousand is 8 --/
theorem min_year_exceed_300k :
  ∀ x : ℤ, (x ≥ 8 ↔ online_shoppers x > 300) ∧
  ∀ y : ℤ, y < 8 → online_shoppers y ≤ 300 :=
sorry


end NUMINAMATH_CALUDE_min_year_exceed_300k_l971_97125


namespace NUMINAMATH_CALUDE_incorrect_to_correct_calculation_l971_97177

theorem incorrect_to_correct_calculation (x : ℝ) : x * 3 - 5 = 103 → (x / 3) - 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_to_correct_calculation_l971_97177


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l971_97105

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l971_97105


namespace NUMINAMATH_CALUDE_complex_point_first_quadrant_l971_97175

theorem complex_point_first_quadrant : 
  let z : ℂ := 1 / Complex.I
  (z^3 + 1).re > 0 ∧ (z^3 + 1).im > 0 := by sorry

end NUMINAMATH_CALUDE_complex_point_first_quadrant_l971_97175


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l971_97113

theorem imaginary_part_of_complex_number (z : ℂ) : z = 1 + 1 / Complex.I → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l971_97113


namespace NUMINAMATH_CALUDE_fraction_calculation_l971_97168

theorem fraction_calculation : 
  (1/5 - 1/3) / ((3/7) / (2/9)) = -28/405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l971_97168


namespace NUMINAMATH_CALUDE_polynomial_identity_l971_97128

-- Define a polynomial function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem polynomial_identity 
  (h : ∀ x : ℝ, f (x^2 + 2) = x^4 + 5*x^2 + 1) :
  ∀ x : ℝ, f (x^2 - 2) = x^4 - 3*x^2 - 3 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l971_97128


namespace NUMINAMATH_CALUDE_same_constant_term_similar_structure_l971_97115

-- Define a polynomial with distinct positive real coefficients
def P (x : ℝ) : ℝ := sorry

-- Define the median of the coefficients of P
def median_coeff : ℝ := sorry

-- Define Q using the median of coefficients of P
def Q (x : ℝ) : ℝ := sorry

-- Theorem stating that P and Q have the same constant term
theorem same_constant_term : P 0 = Q 0 := by sorry

-- Theorem stating that P and Q have similar structure
-- (We can't precisely define "similar structure" without more information,
-- so we'll use a placeholder property)
theorem similar_structure : ∃ (k : ℝ), k > 0 ∧ ∀ x, |P x - Q x| ≤ k := by sorry

end NUMINAMATH_CALUDE_same_constant_term_similar_structure_l971_97115


namespace NUMINAMATH_CALUDE_knitting_productivity_difference_l971_97171

/-- Represents a knitter with their working time and break time -/
structure Knitter where
  workTime : ℕ
  breakTime : ℕ

/-- Calculates the total cycle time for a knitter -/
def cycleTime (k : Knitter) : ℕ := k.workTime + k.breakTime

/-- Calculates the number of cycles in a given time period -/
def cyclesInPeriod (k : Knitter) (period : ℕ) : ℕ :=
  period / cycleTime k

/-- Calculates the total working time in a given period -/
def workingTimeInPeriod (k : Knitter) (period : ℕ) : ℕ :=
  k.workTime * cyclesInPeriod k period

/-- Theorem stating the productivity difference between two knitters -/
theorem knitting_productivity_difference
  (girl1 : Knitter)
  (girl2 : Knitter)
  (h1 : girl1.workTime = 5)
  (h2 : girl2.workTime = 7)
  (h3 : girl1.breakTime = 1)
  (h4 : girl2.breakTime = 1)
  (h5 : ∃ (t : ℕ), workingTimeInPeriod girl1 t = workingTimeInPeriod girl2 t) :
  (workingTimeInPeriod girl2 24 : ℚ) / (workingTimeInPeriod girl1 24 : ℚ) = 21 / 20 := by
  sorry

end NUMINAMATH_CALUDE_knitting_productivity_difference_l971_97171


namespace NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l971_97190

theorem gcd_of_powers_minus_one : 
  Nat.gcd (2^1100 - 1) (2^1122 - 1) = 2^22 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l971_97190


namespace NUMINAMATH_CALUDE_minimum_value_f_l971_97165

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2

theorem minimum_value_f (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ f a 1 ∧ f a 1 = 1) ∨
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ f a (Real.sqrt (-a/2)) ∧
    f a (Real.sqrt (-a/2)) = a/2 * Real.log (-a/2) - a/2 ∧
    -2*(Real.exp 1)^2 < a ∧ a < -2) ∨
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ f a (Real.exp 1) ∧
    f a (Real.exp 1) = a + (Real.exp 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_f_l971_97165


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l971_97180

theorem sqrt_equation_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ Real.sqrt x + 2 * Real.sqrt (x^2 + 7*x) + Real.sqrt (x + 7) = 35 - 2*x ∧ x = 841/144 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l971_97180


namespace NUMINAMATH_CALUDE_checkered_board_division_l971_97199

theorem checkered_board_division (n : ℕ) : 
  (∃ m : ℕ, n^2 = 9 + 7*m) ∧ 
  (∃ k : ℕ, n = 7*k + 3) ↔ 
  n % 7 = 3 :=
sorry

end NUMINAMATH_CALUDE_checkered_board_division_l971_97199


namespace NUMINAMATH_CALUDE_arithmetic_sequence_unique_determination_l971_97197

/-- Given an arithmetic sequence b₁, b₂, b₃, ..., we define:
    S'ₙ = b₁ + b₂ + b₃ + ... + bₙ
    T'ₙ = S'₁ + S'₂ + S'₃ + ... + S'ₙ
    This theorem states that if we know the value of S'₃₀₂₈, 
    then 4543 is the smallest positive integer n for which 
    T'ₙ can be uniquely determined. -/
theorem arithmetic_sequence_unique_determination (b₁ : ℚ) (d : ℚ) (S'₃₀₂₈ : ℚ) :
  let b : ℕ → ℚ := λ n => b₁ + (n - 1) * d
  let S' : ℕ → ℚ := λ n => (n : ℚ) * (2 * b₁ + (n - 1) * d) / 2
  let T' : ℕ → ℚ := λ n => (n * (n + 1) * (3 * b₁ + (n - 1) * d)) / 6
  ∃! (T'₄₅₄₃ : ℚ), S'₃₀₂₈ = S' 3028 ∧ T'₄₅₄₃ = T' 4543 ∧
    ∀ m : ℕ, m < 4543 → ¬∃! (T'ₘ : ℚ), S'₃₀₂₈ = S' 3028 ∧ T'ₘ = T' m :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_unique_determination_l971_97197


namespace NUMINAMATH_CALUDE_smallest_D_value_l971_97141

/-- Represents a three-digit number ABC -/
def threeDigitNumber (A B C : Nat) : Nat :=
  100 * A + 10 * B + C

/-- Represents a four-digit number DCBD -/
def fourDigitNumber (D C B : Nat) : Nat :=
  1000 * D + 100 * C + 10 * B + D

/-- Predicate to check if a number is a single digit -/
def isDigit (n : Nat) : Prop :=
  n ≥ 0 ∧ n ≤ 9

theorem smallest_D_value (A B C D : Nat) :
  isDigit A ∧ isDigit B ∧ isDigit C ∧ isDigit D →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  threeDigitNumber A B C * B = fourDigitNumber D C B →
  ∀ D' : Nat, 
    isDigit D' →
    threeDigitNumber A B C * B = fourDigitNumber D' C B →
    D ≤ D' →
  D = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_D_value_l971_97141


namespace NUMINAMATH_CALUDE_max_value_inequality_l971_97142

theorem max_value_inequality (x : ℝ) (h : x > 1) : 
  (x^2 + 3) / (x - 1) ≥ 6 ∧ 
  ∀ ε > 0, ∃ y > 1, (y^2 + 3) / (y - 1) < 6 + ε :=
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l971_97142


namespace NUMINAMATH_CALUDE_total_questions_l971_97123

/-- Represents the examination structure -/
structure Examination where
  typeA : Nat
  typeB : Nat
  totalTime : Nat
  typeATime : Nat

/-- The given examination parameters -/
def givenExam : Examination where
  typeA := 25
  typeB := 0  -- We don't know this value yet
  totalTime := 180  -- 3 hours * 60 minutes
  typeATime := 40

/-- Theorem stating the total number of questions in the examination -/
theorem total_questions (e : Examination) (h1 : e.typeA = givenExam.typeA)
    (h2 : e.totalTime = givenExam.totalTime) (h3 : e.typeATime = givenExam.typeATime)
    (h4 : 2 * (e.totalTime - e.typeATime) = 7 * e.typeATime) :
    e.typeA + e.typeB = 200 := by
  sorry

#check total_questions

end NUMINAMATH_CALUDE_total_questions_l971_97123


namespace NUMINAMATH_CALUDE_coin_flip_problem_l971_97103

theorem coin_flip_problem : ∃ (n : ℕ+) (a b : ℕ),
  a + b = n ∧
  4 + 8 * a - 3 * b = 1 + 3 * 2^(a - b) ∧
  (4 + 8 * a - 3 * b : ℤ) < 2012 ∧
  n = 137 := by
sorry

end NUMINAMATH_CALUDE_coin_flip_problem_l971_97103


namespace NUMINAMATH_CALUDE_seating_theorem_l971_97189

/-- Represents a taxi with 4 seats -/
structure Taxi :=
  (front_seat : Fin 1)
  (back_seats : Fin 3)

/-- Represents the number of window seats in a taxi -/
def window_seats : Nat := 2

/-- Represents the total number of passengers -/
def total_passengers : Nat := 4

/-- Calculates the number of seating arrangements in a taxi -/
def seating_arrangements (t : Taxi) (w : Nat) (p : Nat) : Nat :=
  w * (p - 1) * (p - 2) * (p - 3)

/-- Theorem stating that the number of seating arrangements is 12 -/
theorem seating_theorem (t : Taxi) :
  seating_arrangements t window_seats total_passengers = 12 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l971_97189


namespace NUMINAMATH_CALUDE_existence_of_finite_sequence_no_infinite_sequence_l971_97194

/-- S(k) denotes the sum of all digits of a positive integer k in its decimal representation. -/
def S (k : ℕ+) : ℕ :=
  sorry

/-- For any positive integer n, there exists an arithmetic sequence of positive integers
    a₁, a₂, ..., aₙ such that S(a₁) < S(a₂) < ... < S(aₙ). -/
theorem existence_of_finite_sequence (n : ℕ+) :
  ∃ (a : ℕ+ → ℕ+) (d : ℕ+), (∀ i : ℕ+, i ≤ n → a i = a 1 + (i - 1) * d) ∧
    (∀ i : ℕ+, i < n → S (a i) < S (a (i + 1))) :=
  sorry

/-- There does not exist an infinite arithmetic sequence of positive integers {aₙ}
    such that S(a₁) < S(a₂) < ... < S(aₙ) < ... -/
theorem no_infinite_sequence :
  ¬ ∃ (a : ℕ+ → ℕ+) (d : ℕ+), (∀ i : ℕ+, a i = a 1 + (i - 1) * d) ∧
    (∀ i j : ℕ+, i < j → S (a i) < S (a j)) :=
  sorry

end NUMINAMATH_CALUDE_existence_of_finite_sequence_no_infinite_sequence_l971_97194


namespace NUMINAMATH_CALUDE_max_crosses_on_10x11_board_l971_97147

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a cross shape -/
structure CrossShape :=
  (size : ℕ := 3)
  (coverage : ℕ := 5)

/-- Defines the maximum number of non-overlapping cross shapes on a chessboard -/
def max_non_overlapping_crosses (board : Chessboard) (cross : CrossShape) : ℕ := sorry

/-- Theorem stating the maximum number of non-overlapping cross shapes on a 10x11 chessboard -/
theorem max_crosses_on_10x11_board :
  ∃ (board : Chessboard) (cross : CrossShape),
    board.rows = 10 ∧ board.cols = 11 ∧
    cross.size = 3 ∧ cross.coverage = 5 ∧
    max_non_overlapping_crosses board cross = 15 := by sorry

end NUMINAMATH_CALUDE_max_crosses_on_10x11_board_l971_97147


namespace NUMINAMATH_CALUDE_students_at_15_44_l971_97182

/-- Represents the number of students in the computer lab at a given time -/
def students_in_lab (initial_students : ℕ) (entry_rate : ℕ) (entry_interval : ℕ) (exit_rate : ℕ) (exit_interval : ℕ) (time : ℕ) : ℕ :=
  initial_students + 
  (time / entry_interval) * entry_rate - 
  ((time - 10) / exit_interval) * exit_rate

/-- Theorem stating the number of students in the lab at 15:44 -/
theorem students_at_15_44 : 
  students_in_lab 20 4 3 8 10 44 = 44 := by
  sorry

end NUMINAMATH_CALUDE_students_at_15_44_l971_97182


namespace NUMINAMATH_CALUDE_system_solution_l971_97173

theorem system_solution : 
  let solutions := [
    (Real.sqrt (2 - Real.sqrt 2) / 2, Real.sqrt (2 + Real.sqrt 2) / 2),
    (Real.sqrt (2 - Real.sqrt 2) / 2, -Real.sqrt (2 + Real.sqrt 2) / 2),
    (-Real.sqrt (2 - Real.sqrt 2) / 2, Real.sqrt (2 + Real.sqrt 2) / 2),
    (-Real.sqrt (2 - Real.sqrt 2) / 2, -Real.sqrt (2 + Real.sqrt 2) / 2),
    (Real.sqrt (2 + Real.sqrt 2) / 2, Real.sqrt (2 - Real.sqrt 2) / 2),
    (Real.sqrt (2 + Real.sqrt 2) / 2, -Real.sqrt (2 - Real.sqrt 2) / 2),
    (-Real.sqrt (2 + Real.sqrt 2) / 2, Real.sqrt (2 - Real.sqrt 2) / 2),
    (-Real.sqrt (2 + Real.sqrt 2) / 2, -Real.sqrt (2 - Real.sqrt 2) / 2)
  ]
  ∀ (x y : ℝ), (x^2 + y^2 = 1 ∧ 4*x*y*(2*y^2 - 1) = 1) ↔ (x, y) ∈ solutions := by
sorry

end NUMINAMATH_CALUDE_system_solution_l971_97173


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l971_97176

theorem complex_magnitude_theorem (x y : ℝ) (z : ℂ) :
  z = x + y * Complex.I →
  (2 * x) / (1 - Complex.I) = 1 + y * Complex.I →
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l971_97176


namespace NUMINAMATH_CALUDE_num_chords_ten_points_l971_97118

/-- The number of chords formed by connecting 2 points out of n points on a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- There are 10 points marked on the circumference of a circle -/
def num_points : ℕ := 10

/-- Theorem: The number of chords formed by connecting 2 points out of 10 points on a circle is 45 -/
theorem num_chords_ten_points : num_chords num_points = 45 := by
  sorry

end NUMINAMATH_CALUDE_num_chords_ten_points_l971_97118


namespace NUMINAMATH_CALUDE_equal_numbers_from_equal_powers_l971_97184

theorem equal_numbers_from_equal_powers (a : Fin 17 → ℕ) 
  (h : ∀ i : Fin 16, (a i) ^ (a (i + 1)) = (a (i + 1)) ^ (a ((i + 2) % 17))) : 
  ∀ i j : Fin 17, a i = a j := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_from_equal_powers_l971_97184


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l971_97124

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space represented by (x - h)^2 + (y - k)^2 = r^2 -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if a point (x, y) is on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line is tangent to a circle -/
def Line.isTangentTo (l : Line) (c : Circle) : Prop :=
  (c.h - l.a * (c.h * l.a + c.k * l.b - l.c) / (l.a^2 + l.b^2))^2 +
  (c.k - l.b * (c.h * l.a + c.k * l.b - l.c) / (l.a^2 + l.b^2))^2 = c.r^2

theorem tangent_lines_to_circle (c : Circle) :
  c.h = 1 ∧ c.k = 0 ∧ c.r = 2 →
  ∃ (l₁ l₂ : Line),
    (l₁.a = 3 ∧ l₁.b = 4 ∧ l₁.c = -13) ∧
    (l₂.a = 1 ∧ l₂.b = 0 ∧ l₂.c = -3) ∧
    l₁.contains 3 1 ∧
    l₂.contains 3 1 ∧
    l₁.isTangentTo c ∧
    l₂.isTangentTo c ∧
    ∀ (l : Line), l.contains 3 1 ∧ l.isTangentTo c → l = l₁ ∨ l = l₂ :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l971_97124


namespace NUMINAMATH_CALUDE_proposition_relation_l971_97109

theorem proposition_relation (a b : ℝ) : 
  (∃ a b : ℝ, |a - b| < 3 ∧ (|a| ≥ 1 ∨ |b| ≥ 2)) ∧
  (∀ a b : ℝ, |a| < 1 ∧ |b| < 2 → |a - b| < 3) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relation_l971_97109


namespace NUMINAMATH_CALUDE_quadratic_roots_bounds_l971_97187

theorem quadratic_roots_bounds (m : ℝ) (x₁ x₂ : ℝ) 
  (hm : m < 0)
  (hroots : x₁^2 - x₁ - 6 = m ∧ x₂^2 - x₂ - 6 = m)
  (horder : x₁ < x₂) :
  -2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_bounds_l971_97187


namespace NUMINAMATH_CALUDE_catriona_fish_count_l971_97172

/-- The number of fish in Catriona's aquarium -/
def total_fish (goldfish angelfish guppies tetras bettas : ℕ) : ℕ :=
  goldfish + angelfish + guppies + tetras + bettas

/-- Theorem stating the total number of fish in Catriona's aquarium -/
theorem catriona_fish_count :
  ∀ (goldfish angelfish guppies tetras bettas : ℕ),
    goldfish = 8 →
    angelfish = goldfish + 4 →
    guppies = 2 * angelfish →
    tetras = goldfish - 3 →
    bettas = tetras + 5 →
    total_fish goldfish angelfish guppies tetras bettas = 59 := by
  sorry

end NUMINAMATH_CALUDE_catriona_fish_count_l971_97172


namespace NUMINAMATH_CALUDE_rotated_rectangle_height_l971_97112

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The configuration of three rectangles with the middle one rotated -/
structure RectangleConfiguration where
  left : Rectangle
  middle : Rectangle
  right : Rectangle
  rotated : Bool

/-- Calculate the height of the top vertex of the middle rectangle when rotated -/
def heightOfRotatedMiddle (config : RectangleConfiguration) : ℝ :=
  if config.rotated then config.middle.width else config.middle.height

/-- The main theorem stating that the height of the rotated middle rectangle is 2 inches -/
theorem rotated_rectangle_height
  (config : RectangleConfiguration)
  (h1 : config.left.width = 2 ∧ config.left.height = 1)
  (h2 : config.middle.width = 2 ∧ config.middle.height = 1)
  (h3 : config.right.width = 2 ∧ config.right.height = 1)
  (h4 : config.rotated = true) :
  heightOfRotatedMiddle config = 2 := by
  sorry

end NUMINAMATH_CALUDE_rotated_rectangle_height_l971_97112


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l971_97110

-- Define a coloring function type
def ColoringFunction := ℝ × ℝ → Bool

-- Define a property for a valid coloring
def ValidColoring (f : ColoringFunction) : Prop :=
  ∀ A B : ℝ × ℝ, A ≠ B →
    ∃ t : ℝ, 0 < t ∧ t < 1 ∧
      let C := (1 - t) • A + t • B
      f C ≠ f A ∨ f C ≠ f B

-- Theorem statement
theorem exists_valid_coloring : ∃ f : ColoringFunction, ValidColoring f := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l971_97110


namespace NUMINAMATH_CALUDE_indeterminate_equation_solutions_l971_97104

def solution_set : Set (ℤ × ℤ) := {(3, -1), (5, 1), (1, 5), (-1, 3)}

theorem indeterminate_equation_solutions :
  {(x, y) : ℤ × ℤ | 2 * (x + y) = x * y + 7} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_indeterminate_equation_solutions_l971_97104
