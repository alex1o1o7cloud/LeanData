import Mathlib

namespace NUMINAMATH_CALUDE_point_B_coordinates_l3089_308999

-- Define the vector type
def Vec := ℝ × ℝ

-- Define point A
def A : Vec := (-1, -5)

-- Define vector a
def a : Vec := (2, 3)

-- Define vector AB in terms of a
def AB : Vec := (3 * a.1, 3 * a.2)

-- Define point B
def B : Vec := (A.1 + AB.1, A.2 + AB.2)

-- Theorem statement
theorem point_B_coordinates : B = (5, 4) := by
  sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l3089_308999


namespace NUMINAMATH_CALUDE_parabola_translation_l3089_308990

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk (1/2) 0 1
  let translated := translate original 1 (-3)
  y = 1/2 * x^2 + 1 → y = 1/2 * (x-1)^2 - 2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_translation_l3089_308990


namespace NUMINAMATH_CALUDE_least_time_four_horses_meet_l3089_308974

def horse_lap_time (k : ℕ) : ℕ := k

def all_horses_lcm : ℕ := 840

theorem least_time_four_horses_meet (T : ℕ) : T = 12 := by
  sorry

end NUMINAMATH_CALUDE_least_time_four_horses_meet_l3089_308974


namespace NUMINAMATH_CALUDE_money_distribution_l3089_308947

theorem money_distribution (total money_ac money_bc : ℕ) 
  (h1 : total = 600)
  (h2 : money_ac = 250)
  (h3 : money_bc = 450) :
  ∃ (a b c : ℕ), a + b + c = total ∧ a + c = money_ac ∧ b + c = money_bc ∧ c = 100 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3089_308947


namespace NUMINAMATH_CALUDE_area_between_circles_l3089_308937

theorem area_between_circles (R r : ℝ) (h : ℝ) : 
  R = 10 → h = 16 → r^2 = R^2 - (h/2)^2 → (R^2 - r^2) * π = 64 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_l3089_308937


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l3089_308920

theorem arithmetic_square_root_of_16 : 
  ∃ x : ℝ, x ≥ 0 ∧ x^2 = 16 ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l3089_308920


namespace NUMINAMATH_CALUDE_function_inequality_l3089_308996

open Real

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h1 : ∀ x : ℝ, f x > deriv f x) : 
  (ℯ^2016 * f (-2016) > f 0) ∧ (f 2016 < ℯ^2016 * f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3089_308996


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l3089_308977

/-- The maximum value of x + 2y for points on the ellipse 2x^2 + 3y^2 = 12 is √22 -/
theorem max_value_on_ellipse :
  ∀ x y : ℝ, 2 * x^2 + 3 * y^2 = 12 →
  ∀ z : ℝ, z = x + 2 * y →
  z ≤ Real.sqrt 22 ∧ ∃ x₀ y₀ : ℝ, 2 * x₀^2 + 3 * y₀^2 = 12 ∧ x₀ + 2 * y₀ = Real.sqrt 22 :=
by sorry


end NUMINAMATH_CALUDE_max_value_on_ellipse_l3089_308977


namespace NUMINAMATH_CALUDE_thursday_to_wednesday_ratio_l3089_308957

/-- Represents the number of laundry loads washed on each day of the week --/
structure LaundryWeek where
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Defines the conditions for Vincent's laundry week --/
def vincentLaundryWeek (w : LaundryWeek) : Prop :=
  w.wednesday = 6 ∧
  w.friday = w.thursday / 2 ∧
  w.saturday = w.wednesday / 3 ∧
  w.wednesday + w.thursday + w.friday + w.saturday = 26

/-- Theorem stating that the ratio of loads washed on Thursday to Wednesday is 2:1 --/
theorem thursday_to_wednesday_ratio (w : LaundryWeek) 
  (h : vincentLaundryWeek w) : w.thursday = 2 * w.wednesday := by
  sorry

end NUMINAMATH_CALUDE_thursday_to_wednesday_ratio_l3089_308957


namespace NUMINAMATH_CALUDE_equation_root_existence_l3089_308963

theorem equation_root_existence (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ ≤ a + b ∧ x₀ = a * Real.sin x₀ + b := by
  sorry

end NUMINAMATH_CALUDE_equation_root_existence_l3089_308963


namespace NUMINAMATH_CALUDE_mixture_qualified_probability_l3089_308951

theorem mixture_qualified_probability 
  (batch1_defective_rate : ℝ)
  (batch2_defective_rate : ℝ)
  (mix_ratio1 : ℝ)
  (mix_ratio2 : ℝ)
  (h1 : batch1_defective_rate = 0.05)
  (h2 : batch2_defective_rate = 0.15)
  (h3 : mix_ratio1 = 3)
  (h4 : mix_ratio2 = 2) :
  let total_ratio := mix_ratio1 + mix_ratio2
  let batch1_qualified_rate := 1 - batch1_defective_rate
  let batch2_qualified_rate := 1 - batch2_defective_rate
  let mixture_qualified_rate := 
    (batch1_qualified_rate * mix_ratio1 + batch2_qualified_rate * mix_ratio2) / total_ratio
  mixture_qualified_rate = 0.91 := by
sorry

end NUMINAMATH_CALUDE_mixture_qualified_probability_l3089_308951


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3089_308985

theorem quadratic_one_root (k : ℝ) : k > 0 ∧ (∃! x : ℝ, x^2 + 6*k*x + 9*k = 0) ↔ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3089_308985


namespace NUMINAMATH_CALUDE_election_combinations_theorem_l3089_308952

/-- Represents a club with members of different genders and ages -/
structure Club where
  total_members : Nat
  girls : Nat
  boys : Nat
  girls_age_order : Fin girls → Nat
  boys_age_order : Fin boys → Nat

/-- Represents the election rules for the club -/
structure ElectionRules where
  president_must_be_girl : Bool
  vp_must_be_boy : Bool
  vp_younger_than_president : Bool

/-- Calculates the number of ways to elect a president and vice-president -/
def election_combinations (club : Club) (rules : ElectionRules) : Nat :=
  sorry

/-- Theorem stating the number of election combinations for the given club and rules -/
theorem election_combinations_theorem (club : Club) (rules : ElectionRules) :
  club.total_members = 25 ∧
  club.girls = 13 ∧
  club.boys = 12 ∧
  rules.president_must_be_girl = true ∧
  rules.vp_must_be_boy = true ∧
  rules.vp_younger_than_president = true →
  election_combinations club rules = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_election_combinations_theorem_l3089_308952


namespace NUMINAMATH_CALUDE_dan_marbles_l3089_308938

/-- The number of marbles Dan has after giving some away -/
def remaining_marbles (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Dan has 96 marbles after giving away 32 from his initial 128 -/
theorem dan_marbles : remaining_marbles 128 32 = 96 := by
  sorry

end NUMINAMATH_CALUDE_dan_marbles_l3089_308938


namespace NUMINAMATH_CALUDE_common_prime_root_quadratics_l3089_308982

theorem common_prime_root_quadratics (a b : ℤ) : 
  (∃ p : ℕ, Prime p ∧ 
    (p : ℤ)^2 + a * p + b = 0 ∧ 
    (p : ℤ)^2 + b * p + 1100 = 0) → 
  a = 274 ∨ a = 40 := by
sorry

end NUMINAMATH_CALUDE_common_prime_root_quadratics_l3089_308982


namespace NUMINAMATH_CALUDE_pentagon_cannot_tile_l3089_308970

-- Define the regular polygons we're considering
inductive RegularPolygon
  | EquilateralTriangle
  | Square
  | Pentagon
  | Hexagon

-- Function to calculate the interior angle of a regular polygon
def interiorAngle (p : RegularPolygon) : ℚ :=
  match p with
  | RegularPolygon.EquilateralTriangle => 60
  | RegularPolygon.Square => 90
  | RegularPolygon.Pentagon => 108
  | RegularPolygon.Hexagon => 120

-- Define what it means for a shape to be able to tile a plane
def canTilePlane (p : RegularPolygon) : Prop :=
  ∃ (n : ℕ), n * interiorAngle p = 360

-- Theorem stating that only the pentagon cannot tile the plane
theorem pentagon_cannot_tile :
  ∀ p : RegularPolygon,
    ¬(canTilePlane p) ↔ p = RegularPolygon.Pentagon :=
by sorry

end NUMINAMATH_CALUDE_pentagon_cannot_tile_l3089_308970


namespace NUMINAMATH_CALUDE_sin_cos_transformation_cos_transformation_sin_cos_equiv_transformation_l3089_308975

theorem sin_cos_transformation (x : Real) : 
  Real.sqrt 2 * Real.sin x = Real.sqrt 2 * Real.cos (x - Real.pi / 2) :=
by sorry

theorem cos_transformation (x : Real) : 
  Real.sqrt 2 * Real.cos (x - Real.pi / 2) = 
  Real.sqrt 2 * Real.cos ((1/2) * (2*x - Real.pi/4) + Real.pi/4) :=
by sorry

theorem sin_cos_equiv_transformation (x : Real) : 
  Real.sqrt 2 * Real.sin x = 
  Real.sqrt 2 * Real.cos ((1/2) * (2*x - Real.pi/4) + Real.pi/4) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_transformation_cos_transformation_sin_cos_equiv_transformation_l3089_308975


namespace NUMINAMATH_CALUDE_transmission_time_approx_seven_minutes_l3089_308921

/-- Represents the data transmission scenario -/
structure DataTransmission where
  num_blocks : ℕ
  chunks_per_block : ℕ
  transmission_rate : ℕ
  delay_per_block : ℕ

/-- Calculates the total transmission time in minutes -/
def total_transmission_time (dt : DataTransmission) : ℚ :=
  let total_chunks := dt.num_blocks * dt.chunks_per_block
  let transmission_time := total_chunks / dt.transmission_rate
  let total_delay := dt.num_blocks * dt.delay_per_block
  (transmission_time + total_delay) / 60

/-- Theorem stating that the transmission time is approximately 7 minutes -/
theorem transmission_time_approx_seven_minutes (dt : DataTransmission) 
  (h1 : dt.num_blocks = 80)
  (h2 : dt.chunks_per_block = 600)
  (h3 : dt.transmission_rate = 150)
  (h4 : dt.delay_per_block = 1) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ |total_transmission_time dt - 7| < ε :=
sorry

end NUMINAMATH_CALUDE_transmission_time_approx_seven_minutes_l3089_308921


namespace NUMINAMATH_CALUDE_inequality_proof_l3089_308929

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2*(a-1)*(b-1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3089_308929


namespace NUMINAMATH_CALUDE_northern_car_speed_l3089_308902

/-- Proves that given the initial conditions of two cars and their movement,
    the speed of the northern car must be 80 mph. -/
theorem northern_car_speed 
  (initial_distance : ℝ) 
  (southern_speed : ℝ) 
  (time : ℝ) 
  (final_distance : ℝ) 
  (h1 : initial_distance = 300) 
  (h2 : southern_speed = 60) 
  (h3 : time = 5) 
  (h4 : final_distance = 500) : 
  ∃ v : ℝ, v = 80 ∧ 
  final_distance^2 = initial_distance^2 + (time * v)^2 + (time * southern_speed)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_northern_car_speed_l3089_308902


namespace NUMINAMATH_CALUDE_tax_rate_on_other_items_l3089_308978

/-- Represents the percentage of total spending on each category -/
structure SpendingPercentages where
  clothing : ℝ
  food : ℝ
  other : ℝ

/-- Represents the tax rates for each category -/
structure TaxRates where
  clothing : ℝ
  food : ℝ
  other : ℝ

/-- Theorem: Given the spending percentages and known tax rates, 
    prove that the tax rate on other items is 8% -/
theorem tax_rate_on_other_items 
  (sp : SpendingPercentages)
  (tr : TaxRates)
  (h1 : sp.clothing = 0.4)
  (h2 : sp.food = 0.3)
  (h3 : sp.other = 0.3)
  (h4 : sp.clothing + sp.food + sp.other = 1)
  (h5 : tr.clothing = 0.04)
  (h6 : tr.food = 0)
  (h7 : sp.clothing * tr.clothing + sp.food * tr.food + sp.other * tr.other = 0.04) :
  tr.other = 0.08 := by
  sorry


end NUMINAMATH_CALUDE_tax_rate_on_other_items_l3089_308978


namespace NUMINAMATH_CALUDE_round_24_6375_to_nearest_tenth_l3089_308948

def round_to_nearest_tenth (x : Float) : Float :=
  (x * 10).round / 10

theorem round_24_6375_to_nearest_tenth :
  round_to_nearest_tenth 24.6375 = 24.6 := by sorry

end NUMINAMATH_CALUDE_round_24_6375_to_nearest_tenth_l3089_308948


namespace NUMINAMATH_CALUDE_max_largest_integer_l3089_308925

theorem max_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℚ) / 5 = 70 →
  e - a = 10 →
  a < b ∧ b < c ∧ c < d ∧ d < e →
  e ≤ 340 :=
sorry

end NUMINAMATH_CALUDE_max_largest_integer_l3089_308925


namespace NUMINAMATH_CALUDE_favorite_subject_count_l3089_308919

theorem favorite_subject_count (total : ℕ) (math_fraction : ℚ) (english_fraction : ℚ)
  (science_fraction : ℚ) (h_total : total = 30) (h_math : math_fraction = 1/5)
  (h_english : english_fraction = 1/3) (h_science : science_fraction = 1/7) :
  total - (total * math_fraction).floor - (total * english_fraction).floor -
  ((total - (total * math_fraction).floor - (total * english_fraction).floor) * science_fraction).floor = 12 :=
by sorry

end NUMINAMATH_CALUDE_favorite_subject_count_l3089_308919


namespace NUMINAMATH_CALUDE_system_solution_unique_l3089_308997

theorem system_solution_unique (x y : ℝ) : 
  (4 * x + 3 * y = 11 ∧ 4 * x - 3 * y = 5) ↔ (x = 2 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3089_308997


namespace NUMINAMATH_CALUDE_distance_to_intersecting_line_l3089_308901

/-- Ellipse G with equation x^2/8 + y^2/4 = 1 -/
def ellipse_G (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

/-- Left focus F1(-2,0) -/
def F1 : ℝ × ℝ := (-2, 0)

/-- Right focus F2(2,0) -/
def F2 : ℝ × ℝ := (2, 0)

/-- Line l intersects ellipse G at points A and B -/
def intersects_ellipse (l : Set (ℝ × ℝ)) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧ A ∈ l ∧ B ∈ l ∧ ellipse_G A.1 A.2 ∧ ellipse_G B.1 B.2

/-- OA is perpendicular to OB -/
def perpendicular (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

/-- Distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: For ellipse G, if line l intersects G at A and B with OA ⊥ OB,
    then the distance from O to l is 2√6/3 -/
theorem distance_to_intersecting_line :
  ∀ l : Set (ℝ × ℝ),
  intersects_ellipse l →
  (∃ A B : ℝ × ℝ, A ≠ B ∧ A ∈ l ∧ B ∈ l ∧ ellipse_G A.1 A.2 ∧ ellipse_G B.1 B.2 ∧ perpendicular A B) →
  distance_point_to_line (0, 0) l = 2 * Real.sqrt 6 / 3 :=
sorry

end NUMINAMATH_CALUDE_distance_to_intersecting_line_l3089_308901


namespace NUMINAMATH_CALUDE_journey_distance_l3089_308949

/-- Proves that a journey with given conditions has a total distance of 112 km -/
theorem journey_distance : 
  ∀ (D : ℝ),
  (D / 2) / 21 + (D / 2) / 24 = 5 →
  D = 112 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l3089_308949


namespace NUMINAMATH_CALUDE_inequality_solution_l3089_308992

theorem inequality_solution (x : ℝ) : 
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔ 
  (x < 1 ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 8) ∨ 10 < x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3089_308992


namespace NUMINAMATH_CALUDE_slope_angles_of_line_l_l3089_308988

/-- Curve C in polar coordinates -/
def curve_C (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

/-- Line l in parametric form -/
def line_l (x y t α : ℝ) : Prop := x = 1 + t * Real.cos α ∧ y = t * Real.sin α

/-- Intersection condition -/
def intersection_condition (t α : ℝ) : Prop := t^2 - 2*t*Real.cos α - 3 = 0

/-- Main theorem -/
theorem slope_angles_of_line_l (α : ℝ) :
  (∃ ρ θ x y t, curve_C ρ θ ∧ line_l x y t α ∧ intersection_condition t α) →
  α = π/4 ∨ α = 3*π/4 :=
sorry

end NUMINAMATH_CALUDE_slope_angles_of_line_l_l3089_308988


namespace NUMINAMATH_CALUDE_no_good_integers_l3089_308998

theorem no_good_integers : 
  ¬∃ (n : ℕ), n ≥ 1 ∧ 
  (∀ (k : ℕ), k > 0 → 
    ((∀ i ∈ Finset.range 9, k % (n + i + 1) = 0) → k % (n + 10) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_no_good_integers_l3089_308998


namespace NUMINAMATH_CALUDE_jeremy_speed_l3089_308941

/-- Given a distance of 20 kilometers and a time of 10 hours, prove that the speed is 2 kilometers per hour. -/
theorem jeremy_speed (distance : ℝ) (time : ℝ) (h1 : distance = 20) (h2 : time = 10) :
  distance / time = 2 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_speed_l3089_308941


namespace NUMINAMATH_CALUDE_percent_relation_l3089_308930

theorem percent_relation (x y : ℝ) (h : (1/2) * (x - y) = (1/5) * (x + y)) :
  y = (3/7) * x := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3089_308930


namespace NUMINAMATH_CALUDE_camryn_practice_schedule_l3089_308909

theorem camryn_practice_schedule :
  let trumpet := 11
  let flute := 3
  let piano := 7
  let violin := 13
  let guitar := 5
  Nat.lcm trumpet (Nat.lcm flute (Nat.lcm piano (Nat.lcm violin guitar))) = 15015 := by
  sorry

end NUMINAMATH_CALUDE_camryn_practice_schedule_l3089_308909


namespace NUMINAMATH_CALUDE_simplify_fraction_l3089_308956

theorem simplify_fraction (a b c : ℕ) (h : b = a * c) :
  (a : ℚ) / b * c = 1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3089_308956


namespace NUMINAMATH_CALUDE_square_perimeter_9cm_l3089_308922

/-- Calculates the perimeter of a square given its side length -/
def square_perimeter (side_length : ℝ) : ℝ := 4 * side_length

/-- Theorem: The perimeter of a square with side length 9 cm is 36 cm -/
theorem square_perimeter_9cm : square_perimeter 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_9cm_l3089_308922


namespace NUMINAMATH_CALUDE_final_number_in_range_l3089_308964

def A : List Nat := List.range (2016 - 672 + 1) |>.map (· + 672)

def replace_step (numbers : List Rat) : List Rat :=
  let (a, b, c) := (numbers.get! 0, numbers.get! 1, numbers.get! 2)
  let new_num := (1 : Rat) / 3 * min a (min b c)
  new_num :: numbers.drop 3

def iterate_replacement (numbers : List Rat) (n : Nat) : List Rat :=
  match n with
  | 0 => numbers
  | n + 1 => iterate_replacement (replace_step numbers) n

theorem final_number_in_range :
  let initial_numbers := A.map (λ x => (x : Rat))
  let final_list := iterate_replacement initial_numbers 672
  final_list.length = 1 ∧ 0 < final_list.head! ∧ final_list.head! < 1 := by
  sorry

end NUMINAMATH_CALUDE_final_number_in_range_l3089_308964


namespace NUMINAMATH_CALUDE_impossible_parking_space_l3089_308989

theorem impossible_parking_space (L W : ℝ) : 
  L = 99 ∧ L + 2 * W = 37 → False :=
by sorry

end NUMINAMATH_CALUDE_impossible_parking_space_l3089_308989


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3089_308942

/-- Given two 2D vectors a and b, where a = (x-1, 2) and b = (2, 1),
    if a is perpendicular to b, then x = 0. -/
theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (x - 1, 2)
  let b : ℝ × ℝ := (2, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 0 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3089_308942


namespace NUMINAMATH_CALUDE_salt_solution_dilution_l3089_308967

def initial_solution_volume : ℝ := 40
def initial_salt_concentration : ℝ := 0.15
def target_salt_concentration : ℝ := 0.10
def water_added : ℝ := 20

theorem salt_solution_dilution :
  let initial_salt_amount : ℝ := initial_solution_volume * initial_salt_concentration
  let final_solution_volume : ℝ := initial_solution_volume + water_added
  let final_salt_concentration : ℝ := initial_salt_amount / final_solution_volume
  final_salt_concentration = target_salt_concentration := by sorry

end NUMINAMATH_CALUDE_salt_solution_dilution_l3089_308967


namespace NUMINAMATH_CALUDE_max_amount_received_back_l3089_308926

/-- Represents the denominations of chips --/
inductive ChipDenomination
  | twoHundred
  | fiveHundred

/-- Represents the number of chips lost for each denomination --/
structure ChipsLost where
  twoHundred : ℕ
  fiveHundred : ℕ

def totalChipsBought : ℕ := 50000

def chipValue (d : ChipDenomination) : ℕ :=
  match d with
  | ChipDenomination.twoHundred => 200
  | ChipDenomination.fiveHundred => 500

def totalChipsLost (c : ChipsLost) : ℕ := c.twoHundred + c.fiveHundred

def validChipsLost (c : ChipsLost) : Prop :=
  totalChipsLost c = 25 ∧
  (c.twoHundred = c.fiveHundred + 5 ∨ c.twoHundred + 5 = c.fiveHundred)

def valueLost (c : ChipsLost) : ℕ :=
  c.twoHundred * chipValue ChipDenomination.twoHundred +
  c.fiveHundred * chipValue ChipDenomination.fiveHundred

def amountReceivedBack (c : ChipsLost) : ℕ := totalChipsBought - valueLost c

theorem max_amount_received_back :
  ∃ (c : ChipsLost), validChipsLost c ∧
    (∀ (c' : ChipsLost), validChipsLost c' → amountReceivedBack c ≥ amountReceivedBack c') ∧
    amountReceivedBack c = 42000 :=
  sorry

end NUMINAMATH_CALUDE_max_amount_received_back_l3089_308926


namespace NUMINAMATH_CALUDE_continuous_piecewise_sum_l3089_308914

/-- A piecewise function f(x) defined on the real line. -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then a * x + 6
  else if x ≥ -3 then x - 7
  else 3 * x - b

/-- The function f is continuous on the real line. -/
def is_continuous (a b : ℝ) : Prop :=
  Continuous (f a b)

/-- If f is continuous, then a + b = -7/3. -/
theorem continuous_piecewise_sum (a b : ℝ) :
  is_continuous a b → a + b = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_continuous_piecewise_sum_l3089_308914


namespace NUMINAMATH_CALUDE_baseball_earnings_l3089_308933

/-- The total earnings from two baseball games -/
def total_earnings (saturday_earnings wednesday_earnings : ℚ) : ℚ :=
  saturday_earnings + wednesday_earnings

/-- Theorem stating the total earnings from two baseball games -/
theorem baseball_earnings : 
  ∃ (saturday_earnings wednesday_earnings : ℚ),
    saturday_earnings = 2662.50 ∧
    wednesday_earnings = saturday_earnings - 142.50 ∧
    total_earnings saturday_earnings wednesday_earnings = 5182.50 :=
by sorry

end NUMINAMATH_CALUDE_baseball_earnings_l3089_308933


namespace NUMINAMATH_CALUDE_log_sum_equals_three_l3089_308994

theorem log_sum_equals_three : Real.log 4 / Real.log 10 + Real.log 25 / Real.log 10 + (-1/8)^0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_three_l3089_308994


namespace NUMINAMATH_CALUDE_water_amount_is_150_l3089_308946

/-- Represents the ratios of bleach, detergent, and water in a solution --/
structure SolutionRatio where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- The original ratio of the solution --/
def original_ratio : SolutionRatio := ⟨4, 40, 100⟩

/-- The altered ratio after tripling bleach to detergent and halving detergent to water --/
def altered_ratio : SolutionRatio :=
  let b := original_ratio.bleach * 3
  let d := original_ratio.detergent
  let w := original_ratio.water / 2
  ⟨b, d, w⟩

/-- The amount of detergent in the altered solution --/
def altered_detergent_amount : ℚ := 60

/-- Calculates the amount of water in the altered solution --/
def water_amount : ℚ :=
  altered_detergent_amount * (altered_ratio.water / altered_ratio.detergent)

/-- Theorem stating that the amount of water in the altered solution is 150 liters --/
theorem water_amount_is_150 : water_amount = 150 := by sorry

end NUMINAMATH_CALUDE_water_amount_is_150_l3089_308946


namespace NUMINAMATH_CALUDE_bus_seat_capacity_l3089_308916

/-- Represents the capacity of a bus with specific seating arrangements. -/
structure BusCapacity where
  left_seats : ℕ
  right_seats : ℕ
  back_seat_capacity : ℕ
  total_capacity : ℕ

/-- Calculates the number of people each regular seat can hold. -/
def seats_capacity (bus : BusCapacity) : ℚ :=
  (bus.total_capacity - bus.back_seat_capacity) / (bus.left_seats + bus.right_seats)

/-- Theorem stating that for the given bus configuration, each seat can hold 3 people. -/
theorem bus_seat_capacity :
  let bus := BusCapacity.mk 15 12 9 90
  seats_capacity bus = 3 := by sorry

end NUMINAMATH_CALUDE_bus_seat_capacity_l3089_308916


namespace NUMINAMATH_CALUDE_integral_symmetric_function_l3089_308924

theorem integral_symmetric_function (a : ℝ) (h : a > 0) :
  ∫ x in -a..a, (x^2 * Real.cos x + Real.exp x) / (Real.exp x + 1) = a := by sorry

end NUMINAMATH_CALUDE_integral_symmetric_function_l3089_308924


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l3089_308973

theorem solution_set_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, |a * x + 2| < 6 ↔ -1 < x ∧ x < 2) → a = -4 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l3089_308973


namespace NUMINAMATH_CALUDE_smallest_ccd_value_l3089_308976

theorem smallest_ccd_value (C D : ℕ) : 
  (1 ≤ C ∧ C ≤ 9) →
  (1 ≤ D ∧ D ≤ 9) →
  C ≠ D →
  (10 * C + D : ℕ) < 100 →
  (100 * C + 10 * C + D : ℕ) < 1000 →
  (10 * C + D : ℕ) = (100 * C + 10 * C + D : ℕ) / 7 →
  (∀ (C' D' : ℕ), 
    (1 ≤ C' ∧ C' ≤ 9) →
    (1 ≤ D' ∧ D' ≤ 9) →
    C' ≠ D' →
    (10 * C' + D' : ℕ) < 100 →
    (100 * C' + 10 * C' + D' : ℕ) < 1000 →
    (10 * C' + D' : ℕ) = (100 * C' + 10 * C' + D' : ℕ) / 7 →
    (100 * C + 10 * C + D : ℕ) ≤ (100 * C' + 10 * C' + D' : ℕ)) →
  (100 * C + 10 * C + D : ℕ) = 115 :=
by sorry

end NUMINAMATH_CALUDE_smallest_ccd_value_l3089_308976


namespace NUMINAMATH_CALUDE_lunch_solution_l3089_308928

def lunch_problem (total_spent : ℕ) (friend_spent : ℕ) : Prop :=
  friend_spent > total_spent - friend_spent ∧
  friend_spent - (total_spent - friend_spent) = 3

theorem lunch_solution :
  lunch_problem 19 11 := by sorry

end NUMINAMATH_CALUDE_lunch_solution_l3089_308928


namespace NUMINAMATH_CALUDE_four_thirds_of_number_is_36_l3089_308918

theorem four_thirds_of_number_is_36 (x : ℚ) : (4 : ℚ) / 3 * x = 36 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_four_thirds_of_number_is_36_l3089_308918


namespace NUMINAMATH_CALUDE_davids_math_marks_l3089_308945

/-- Represents the marks obtained in each subject -/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average marks -/
def average (m : Marks) : ℚ :=
  (m.english + m.mathematics + m.physics + m.chemistry + m.biology) / 5

/-- Theorem: Given David's marks in other subjects and his average, his Mathematics marks must be 65 -/
theorem davids_math_marks (m : Marks) : 
  m.english = 51 → 
  m.physics = 82 → 
  m.chemistry = 67 → 
  m.biology = 85 → 
  average m = 70 → 
  m.mathematics = 65 := by
  sorry

#eval average { english := 51, mathematics := 65, physics := 82, chemistry := 67, biology := 85 }

end NUMINAMATH_CALUDE_davids_math_marks_l3089_308945


namespace NUMINAMATH_CALUDE_apple_to_cucumber_ratio_l3089_308953

/-- Given the cost ratios of fruits, calculate the equivalent number of cucumbers for 20 apples -/
theorem apple_to_cucumber_ratio 
  (apple_banana_ratio : ℚ) 
  (banana_cucumber_ratio : ℚ) 
  (h1 : apple_banana_ratio = 10 / 5)  -- 10 apples = 5 bananas
  (h2 : banana_cucumber_ratio = 3 / 4)  -- 3 bananas = 4 cucumbers
  : (20 : ℚ) / apple_banana_ratio * banana_cucumber_ratio⁻¹ = 40 / 3 :=
by sorry

end NUMINAMATH_CALUDE_apple_to_cucumber_ratio_l3089_308953


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_plus_one_choose_n_l3089_308940

theorem binomial_coefficient_n_plus_one_choose_n (n : ℕ+) : 
  Nat.choose (n + 1) n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_plus_one_choose_n_l3089_308940


namespace NUMINAMATH_CALUDE_last_two_digits_squared_l3089_308972

theorem last_two_digits_squared (n : ℤ) : 
  (n * 402 * 503 * 604 * 646 * 547 * 448 * 349) ^ 2 % 100 = 76 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_squared_l3089_308972


namespace NUMINAMATH_CALUDE_point_translation_l3089_308993

def initial_point : ℝ × ℝ := (-5, 1)
def x_translation : ℝ := 2
def y_translation : ℝ := -4

theorem point_translation (P : ℝ × ℝ) (dx dy : ℝ) :
  P = initial_point →
  (P.1 + dx, P.2 + dy) = (-3, -3) :=
by sorry

end NUMINAMATH_CALUDE_point_translation_l3089_308993


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_is_14pi_l3089_308983

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a right triangle with side lengths 3, 4, and 5 -/
structure RightTriangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ
  side_ab : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 3
  side_bc : Real.sqrt ((b.1 - c.1)^2 + (b.2 - c.2)^2) = 4
  side_ca : Real.sqrt ((c.1 - a.1)^2 + (c.2 - a.2)^2) = 5
  right_angle : (a.1 - b.1) * (c.1 - b.1) + (a.2 - b.2) * (c.2 - b.2) = 0

/-- Checks if two circles are externally tangent -/
def areExternallyTangent (c1 c2 : Circle) : Prop :=
  Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = c1.radius + c2.radius

/-- Theorem: The sum of the areas of three mutually externally tangent circles
    centered at the vertices of a 3-4-5 right triangle is 14π -/
theorem sum_of_circle_areas_is_14pi (t : RightTriangle) 
    (c1 : Circle) (c2 : Circle) (c3 : Circle)
    (h1 : c1.center = t.a) (h2 : c2.center = t.b) (h3 : c3.center = t.c)
    (h4 : areExternallyTangent c1 c2)
    (h5 : areExternallyTangent c2 c3)
    (h6 : areExternallyTangent c3 c1) :
    π * (c1.radius^2 + c2.radius^2 + c3.radius^2) = 14 * π := by
  sorry


end NUMINAMATH_CALUDE_sum_of_circle_areas_is_14pi_l3089_308983


namespace NUMINAMATH_CALUDE_systematic_sampling_first_number_l3089_308931

/-- Systematic sampling problem -/
theorem systematic_sampling_first_number
  (population : ℕ)
  (sample_size : ℕ)
  (eighteenth_sample : ℕ)
  (h1 : population = 1000)
  (h2 : sample_size = 40)
  (h3 : eighteenth_sample = 443)
  (h4 : sample_size > 0)
  (h5 : population ≥ sample_size) :
  ∃ (first_sample : ℕ),
    first_sample + 17 * (population / sample_size) = eighteenth_sample ∧
    first_sample = 18 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_first_number_l3089_308931


namespace NUMINAMATH_CALUDE_expression_evaluation_l3089_308936

theorem expression_evaluation (x : ℚ) (h : x = -4) : 
  (1 - 4 / (x + 3)) / ((x^2 - 1) / (x^2 + 6*x + 9)) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3089_308936


namespace NUMINAMATH_CALUDE_real_z_implies_m_eq_3_modulus_z_eq_sqrt_13_when_m_eq_1_l3089_308908

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m + 2) (m - 3)

-- Theorem 1: If z is a real number, then m = 3
theorem real_z_implies_m_eq_3 (m : ℝ) : z m = Complex.mk (z m).re 0 → m = 3 := by
  sorry

-- Theorem 2: When m = 1, the modulus of z is √13
theorem modulus_z_eq_sqrt_13_when_m_eq_1 : Complex.abs (z 1) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_real_z_implies_m_eq_3_modulus_z_eq_sqrt_13_when_m_eq_1_l3089_308908


namespace NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l3089_308912

/-- Given three positive real numbers forming an arithmetic sequence,
    the sum of their ratio and its reciprocal is at least 5/2 -/
theorem min_value_arithmetic_sequence (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_arith : b - a = c - b) : (a + c) / b + b / (a + c) ≥ 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l3089_308912


namespace NUMINAMATH_CALUDE_triangle_abc_area_l3089_308969

/-- Triangle ABC with vertices A(0,0), B(1,7), and C(0,8) has an area of 28 square units -/
theorem triangle_abc_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 7)
  let C : ℝ × ℝ := (0, 8)
  let triangle_area := (1/2) * |(C.2 - A.2)| * |(B.1 - A.1)|
  triangle_area = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_area_l3089_308969


namespace NUMINAMATH_CALUDE_circular_garden_radius_l3089_308968

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1 / 8) * π * r^2 → r = 16 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l3089_308968


namespace NUMINAMATH_CALUDE_work_completion_time_l3089_308987

theorem work_completion_time (a b c total : ℚ) : 
  a > 0 → b > 0 → c > 0 → total > 0 →
  (1 / a + 1 / b + 1 / c = 1 / total) →
  (1 / 8 + 1 / 12 + 1 / c = 1 / 4) →
  c = 24 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3089_308987


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3089_308935

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if its focal length is 10 and the point (1, 2) lies on its asymptote,
    then its equation is x²/5 - y²/20 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2 = 25) →
  (b - 2*a = 0) →
  ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2/5 - y^2/20 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3089_308935


namespace NUMINAMATH_CALUDE_stamp_exhibition_l3089_308958

theorem stamp_exhibition (people : ℕ) (total_stamps : ℕ) : 
  (3 * people + 24 = total_stamps) →
  (4 * people = total_stamps + 26) →
  total_stamps = 174 := by
sorry

end NUMINAMATH_CALUDE_stamp_exhibition_l3089_308958


namespace NUMINAMATH_CALUDE_shirt_price_l3089_308903

theorem shirt_price (total_cost sweater_price shirt_price : ℝ) : 
  total_cost = 80.34 →
  shirt_price = sweater_price - 7.43 →
  total_cost = sweater_price + shirt_price →
  shirt_price = 36.455 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_l3089_308903


namespace NUMINAMATH_CALUDE_min_value_expression_l3089_308913

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (min : ℝ), min = Real.sqrt 6 ∧
  ∀ (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0),
    x^2 + 2*y^2 + 1/x^2 + 2*y/x ≥ min ∧
    ∃ (a₀ b₀ : ℝ) (ha₀ : a₀ ≠ 0) (hb₀ : b₀ ≠ 0),
      a₀^2 + 2*b₀^2 + 1/a₀^2 + 2*b₀/a₀ = min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3089_308913


namespace NUMINAMATH_CALUDE_f_composition_negative_two_l3089_308932

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

-- Theorem statement
theorem f_composition_negative_two (f : ℝ → ℝ) :
  (∀ x, x ≥ 0 → f x = 1 - Real.sqrt x) →
  (∀ x, x < 0 → f x = 2^x) →
  f (f (-2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_two_l3089_308932


namespace NUMINAMATH_CALUDE_range_of_a_l3089_308980

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- Define the theorem
theorem range_of_a (a : ℝ) : A ∪ B a = B a → -1 < a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3089_308980


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_problem_solution_l3089_308917

theorem fraction_of_fraction_of_fraction (n : ℕ) : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * n = n / 36 := by
  sorry

theorem problem_solution : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 180 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_problem_solution_l3089_308917


namespace NUMINAMATH_CALUDE_left_handed_classical_music_lovers_l3089_308915

theorem left_handed_classical_music_lovers (total : ℕ) (left_handed : ℕ) (classical_music : ℕ) (right_handed_non_classical : ℕ) 
  (h1 : total = 25)
  (h2 : left_handed = 10)
  (h3 : classical_music = 18)
  (h4 : right_handed_non_classical = 3)
  (h5 : left_handed + (total - left_handed) = total) :
  ∃ y : ℕ, y = 6 ∧ 
    y + (left_handed - y) + (classical_music - y) + right_handed_non_classical = total :=
by sorry

end NUMINAMATH_CALUDE_left_handed_classical_music_lovers_l3089_308915


namespace NUMINAMATH_CALUDE_box_packing_problem_l3089_308962

theorem box_packing_problem (x y : ℤ) : 
  (3 * x + 4 * y = 108) → 
  (2 * x + 3 * y = 76) → 
  (x = 20 ∧ y = 12) := by
sorry

end NUMINAMATH_CALUDE_box_packing_problem_l3089_308962


namespace NUMINAMATH_CALUDE_roger_step_goal_time_l3089_308905

/-- Represents the number of steps Roger can walk in 30 minutes -/
def steps_per_30_min : ℕ := 2000

/-- Represents Roger's daily step goal -/
def daily_goal : ℕ := 10000

/-- Represents the time in minutes it takes Roger to reach his daily goal -/
def time_to_reach_goal : ℕ := 150

/-- Theorem stating that the time required for Roger to reach his daily goal is 150 minutes -/
theorem roger_step_goal_time : 
  (daily_goal / steps_per_30_min) * 30 = time_to_reach_goal :=
by sorry

end NUMINAMATH_CALUDE_roger_step_goal_time_l3089_308905


namespace NUMINAMATH_CALUDE_smallest_sum_A_plus_b_l3089_308979

theorem smallest_sum_A_plus_b : 
  ∀ (A : ℕ) (b : ℕ),
    A < 4 →
    A > 0 →
    b > 5 →
    21 * A = 3 * b + 3 →
    ∀ (A' : ℕ) (b' : ℕ),
      A' < 4 →
      A' > 0 →
      b' > 5 →
      21 * A' = 3 * b' + 3 →
      A + b ≤ A' + b' :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_A_plus_b_l3089_308979


namespace NUMINAMATH_CALUDE_statement_two_is_false_l3089_308910

/-- Definition of the heart operation -/
def heart (x y : ℝ) : ℝ := 2 * |x - y| + 1

/-- Theorem stating that Statement 2 is false -/
theorem statement_two_is_false :
  ∃ x y : ℝ, 3 * (heart x y) ≠ heart (3 * x) (3 * y) :=
sorry

end NUMINAMATH_CALUDE_statement_two_is_false_l3089_308910


namespace NUMINAMATH_CALUDE_sin_cos_cube_difference_l3089_308961

theorem sin_cos_cube_difference (α : ℝ) (n : ℝ) (h : Real.sin α - Real.cos α = n) :
  Real.sin α ^ 3 - Real.cos α ^ 3 = (3 * n - n^3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_cube_difference_l3089_308961


namespace NUMINAMATH_CALUDE_a_plus_b_eq_neg_one_l3089_308906

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {1, a, b}
def B (a : ℝ) : Set ℝ := {a, a^2, a*a}

-- State the theorem
theorem a_plus_b_eq_neg_one (a b : ℝ) : A a b = B a → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_eq_neg_one_l3089_308906


namespace NUMINAMATH_CALUDE_three_couples_arrangement_l3089_308907

/-- The number of arrangements for three couples standing in a row -/
def couple_arrangements : ℕ := 48

/-- The number of ways to arrange three distinct units in a row -/
def unit_arrangements : ℕ := 6

/-- The number of internal arrangements for each couple -/
def internal_arrangements : ℕ := 2

/-- Theorem: The number of different arrangements for three couples standing in a row,
    where each couple must stand next to each other, is equal to 48. -/
theorem three_couples_arrangement :
  couple_arrangements = unit_arrangements * internal_arrangements^3 :=
by sorry

end NUMINAMATH_CALUDE_three_couples_arrangement_l3089_308907


namespace NUMINAMATH_CALUDE_train_speed_increase_l3089_308959

theorem train_speed_increase (distance : ℝ) (speed_increase : ℝ) (time_reduction : ℝ) (speed_limit : ℝ)
  (h1 : distance = 1600)
  (h2 : speed_increase = 20)
  (h3 : time_reduction = 4)
  (h4 : speed_limit = 140) :
  ∃ (original_speed : ℝ),
    original_speed > 0 ∧
    distance / original_speed = distance / (original_speed + speed_increase) + time_reduction ∧
    original_speed + speed_increase < speed_limit :=
by sorry

#check train_speed_increase

end NUMINAMATH_CALUDE_train_speed_increase_l3089_308959


namespace NUMINAMATH_CALUDE_min_income_2020_l3089_308923

/-- Represents the per capita income growth over a 40-year period -/
def income_growth (initial : ℝ) (mid : ℝ) (final : ℝ) : Prop :=
  ∃ (x : ℝ), 
    initial * (1 + x)^20 ≥ mid ∧
    initial * (1 + x)^40 ≥ final

/-- Theorem stating the minimum per capita income in 2020 based on 1980 and 2000 data -/
theorem min_income_2020 : income_growth 250 800 2560 := by
  sorry

end NUMINAMATH_CALUDE_min_income_2020_l3089_308923


namespace NUMINAMATH_CALUDE_house_area_proof_l3089_308991

def house_painting_problem (price_per_sqft : ℝ) (total_cost : ℝ) : Prop :=
  price_per_sqft > 0 ∧ total_cost > 0 ∧ (total_cost / price_per_sqft = 88)

theorem house_area_proof :
  house_painting_problem 20 1760 :=
sorry

end NUMINAMATH_CALUDE_house_area_proof_l3089_308991


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l3089_308995

theorem smallest_prime_dividing_sum : ∃ p : Nat, 
  Prime p ∧ 
  p ∣ (4^15 + 7^12) ∧ 
  ∀ q : Nat, Prime q → q ∣ (4^15 + 7^12) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l3089_308995


namespace NUMINAMATH_CALUDE_expression_equals_two_l3089_308927

theorem expression_equals_two : 
  (Real.sqrt 3) ^ 0 + 2⁻¹ + Real.sqrt 2 * Real.cos (45 * π / 180) - |-(1/2)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l3089_308927


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3089_308911

def A : Set Int := {-1, 1, 2, 4}
def B : Set Int := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3089_308911


namespace NUMINAMATH_CALUDE_hexagon_area_l3089_308950

/-- Regular hexagon with vertices A at (0,0) and C at (10,2) -/
structure RegularHexagon where
  A : ℝ × ℝ
  C : ℝ × ℝ
  is_regular : Bool
  A_is_origin : A = (0, 0)
  C_coordinates : C = (10, 2)

/-- The area of a regular hexagon -/
def area (h : RegularHexagon) : ℝ := sorry

/-- Theorem stating that the area of the specified regular hexagon is 52√3 -/
theorem hexagon_area (h : RegularHexagon) : area h = 52 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_l3089_308950


namespace NUMINAMATH_CALUDE_chocolate_distribution_l3089_308984

/-- Proves that each friend receives 24/7 pounds of chocolate given the initial conditions -/
theorem chocolate_distribution (total : ℚ) (initial_piles : ℕ) (friends : ℕ) : 
  total = 60 / 7 →
  initial_piles = 5 →
  friends = 2 →
  (total - (total / initial_piles)) / friends = 24 / 7 := by
sorry

#eval (60 / 7 : ℚ)
#eval (24 / 7 : ℚ)

end NUMINAMATH_CALUDE_chocolate_distribution_l3089_308984


namespace NUMINAMATH_CALUDE_complete_square_constant_l3089_308965

theorem complete_square_constant (x : ℝ) :
  ∃ (a h k : ℝ), x^2 - 8*x = a*(x - h)^2 + k ∧ k = -16 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_constant_l3089_308965


namespace NUMINAMATH_CALUDE_fine_amount_correct_l3089_308955

/-- Calculates the fine amount given the quantity sold, price per ounce, and amount left after the fine -/
def calculate_fine (quantity_sold : ℕ) (price_per_ounce : ℕ) (amount_left : ℕ) : ℕ :=
  quantity_sold * price_per_ounce - amount_left

/-- Proves that the fine amount is correct given the problem conditions -/
theorem fine_amount_correct : calculate_fine 8 9 22 = 50 := by
  sorry

end NUMINAMATH_CALUDE_fine_amount_correct_l3089_308955


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3089_308934

theorem complex_fraction_simplification (x y z : ℚ) 
  (hx : x = 4)
  (hy : y = 5)
  (hz : z = 2) :
  (1 / z / y) / (1 / x) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3089_308934


namespace NUMINAMATH_CALUDE_twenty_first_term_equals_203_l3089_308900

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1) * seq.common_difference

theorem twenty_first_term_equals_203 :
  ∃ (seq : ArithmeticSequence),
    seq.first_term = 3 ∧
    nth_term seq 2 = 13 ∧
    nth_term seq 3 = 23 ∧
    nth_term seq 21 = 203 := by
  sorry

end NUMINAMATH_CALUDE_twenty_first_term_equals_203_l3089_308900


namespace NUMINAMATH_CALUDE_binomial_12_9_l3089_308966

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_9_l3089_308966


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3089_308960

theorem square_plus_reciprocal_square (x : ℝ) (h : x ≠ 0) :
  x^2 + 1/x^2 = 2 → x^4 + 1/x^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3089_308960


namespace NUMINAMATH_CALUDE_new_alcohol_concentration_l3089_308944

/-- Represents a vessel containing an alcohol mixture -/
structure Vessel where
  capacity : ℝ
  alcohol_concentration : ℝ

/-- Calculates the amount of alcohol in a vessel -/
def alcohol_amount (v : Vessel) : ℝ := v.capacity * v.alcohol_concentration

theorem new_alcohol_concentration
  (vessel1 : Vessel)
  (vessel2 : Vessel)
  (final_capacity : ℝ)
  (h1 : vessel1.capacity = 2)
  (h2 : vessel1.alcohol_concentration = 0.4)
  (h3 : vessel2.capacity = 6)
  (h4 : vessel2.alcohol_concentration = 0.6)
  (h5 : final_capacity = 10)
  (h6 : vessel1.capacity + vessel2.capacity = 8) :
  let total_alcohol := alcohol_amount vessel1 + alcohol_amount vessel2
  let new_concentration := total_alcohol / final_capacity
  new_concentration = 0.44 := by
  sorry

#check new_alcohol_concentration

end NUMINAMATH_CALUDE_new_alcohol_concentration_l3089_308944


namespace NUMINAMATH_CALUDE_largest_solution_value_l3089_308981

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.log 10 / Real.log (x^2) + Real.log 10 / Real.log (x^4) + Real.log 10 / Real.log (9*x^5) = 0

-- Define the set of solutions
def solution_set := { x : ℝ | equation x ∧ x > 0 }

-- State the theorem
theorem largest_solution_value :
  ∃ (x : ℝ), x ∈ solution_set ∧ 
  (∀ (y : ℝ), y ∈ solution_set → y ≤ x) ∧
  (1 / x^18 = 9^93) := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_value_l3089_308981


namespace NUMINAMATH_CALUDE_matrix_zero_product_implies_zero_multiplier_l3089_308971

theorem matrix_zero_product_implies_zero_multiplier 
  (A B : Matrix (Fin 3) (Fin 3) ℂ) 
  (hB : B ≠ 0) 
  (hAB : A * B = 0) : 
  ∃ D : Matrix (Fin 3) (Fin 3) ℂ, D ≠ 0 ∧ A * D = 0 ∧ D * A = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_zero_product_implies_zero_multiplier_l3089_308971


namespace NUMINAMATH_CALUDE_special_isosceles_inscribed_circle_radius_l3089_308943

/-- An isosceles triangle with a specific inscribed circle property -/
structure SpecialIsoscelesTriangle where
  -- Base of the triangle
  base : ℝ
  -- Ratio of the parts of the altitude divided by the center of the inscribed circle
  altitude_ratio : ℝ × ℝ
  -- The triangle is isosceles
  isIsosceles : True
  -- The base is 60
  base_is_60 : base = 60
  -- The ratio is 17:15
  ratio_is_17_15 : altitude_ratio = (17, 15)

/-- The radius of the inscribed circle in the special isosceles triangle -/
def inscribed_circle_radius (t : SpecialIsoscelesTriangle) : ℝ := 7.5

/-- Theorem: The radius of the inscribed circle in the special isosceles triangle is 7.5 -/
theorem special_isosceles_inscribed_circle_radius (t : SpecialIsoscelesTriangle) :
  inscribed_circle_radius t = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_special_isosceles_inscribed_circle_radius_l3089_308943


namespace NUMINAMATH_CALUDE_geometric_sum_five_terms_l3089_308904

theorem geometric_sum_five_terms :
  (1 / 5 : ℚ) - (1 / 25 : ℚ) + (1 / 125 : ℚ) - (1 / 625 : ℚ) + (1 / 3125 : ℚ) = 521 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_five_terms_l3089_308904


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l3089_308986

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l3089_308986


namespace NUMINAMATH_CALUDE_sam_speed_l3089_308954

/-- Represents a point on the route --/
structure Point where
  position : ℝ

/-- Represents a person traveling on the route --/
structure Traveler where
  start : Point
  speed : ℝ

/-- The scenario of Sam and Nik's travel --/
structure TravelScenario where
  sam : Traveler
  nik : Traveler
  meetingPoint : Point
  totalTime : ℝ

/-- The given travel scenario --/
def givenScenario : TravelScenario where
  sam := { start := { position := 0 }, speed := 0 }  -- speed will be calculated
  nik := { start := { position := 1000 }, speed := 0 }  -- speed is not needed for the problem
  meetingPoint := { position := 600 }
  totalTime := 20

theorem sam_speed (scenario : TravelScenario) :
  scenario.sam.start.position = 0 ∧
  scenario.nik.start.position = 1000 ∧
  scenario.meetingPoint.position = 600 ∧
  scenario.totalTime = 20 →
  scenario.sam.speed = 50 := by
  sorry

end NUMINAMATH_CALUDE_sam_speed_l3089_308954


namespace NUMINAMATH_CALUDE_exists_positive_decreasing_function_l3089_308939

theorem exists_positive_decreasing_function :
  ∃ f : ℝ → ℝ, (∀ x y : ℝ, x < y → f y < f x) ∧ (∀ x : ℝ, f x > 0) := by
  sorry

end NUMINAMATH_CALUDE_exists_positive_decreasing_function_l3089_308939
