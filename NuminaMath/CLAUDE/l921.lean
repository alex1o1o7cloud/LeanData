import Mathlib

namespace NUMINAMATH_CALUDE_counterexample_exists_l921_92185

theorem counterexample_exists : ∃ (a b c d : ℝ), 
  ((a + b) / (3*a - b) = (b + c) / (3*b - c)) ∧
  ((b + c) / (3*b - c) = (c + d) / (3*c - d)) ∧
  ((c + d) / (3*c - d) = (d + a) / (3*d - a)) ∧
  (3*a - b ≠ 0) ∧ (3*b - c ≠ 0) ∧ (3*c - d ≠ 0) ∧ (3*d - a ≠ 0) ∧
  (a^2 + b^2 + c^2 + d^2 ≠ a*b + b*c + c*d + d*a) :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l921_92185


namespace NUMINAMATH_CALUDE_circle_equation_radius_l921_92176

theorem circle_equation_radius (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 4*y - k = 0 ↔ (x + 4)^2 + (y + 2)^2 = 49) ↔ 
  k = 29 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_radius_l921_92176


namespace NUMINAMATH_CALUDE_common_root_quadratic_equations_l921_92192

theorem common_root_quadratic_equations (p : ℝ) :
  (∃ x : ℝ, x^2 - (p+2)*x + 2*p + 6 = 0 ∧ 2*x^2 - (p+4)*x + 2*p + 3 = 0) ↔
  (p = -3 ∨ p = 9) ∧
  ((p = -3 → ∃ x : ℝ, x = -1 ∧ x^2 - (p+2)*x + 2*p + 6 = 0 ∧ 2*x^2 - (p+4)*x + 2*p + 3 = 0) ∧
   (p = 9 → ∃ x : ℝ, x = 3 ∧ x^2 - (p+2)*x + 2*p + 6 = 0 ∧ 2*x^2 - (p+4)*x + 2*p + 3 = 0)) :=
by sorry

#check common_root_quadratic_equations

end NUMINAMATH_CALUDE_common_root_quadratic_equations_l921_92192


namespace NUMINAMATH_CALUDE_tower_lights_problem_l921_92110

theorem tower_lights_problem (n : ℕ) (r : ℝ) (sum : ℝ) (h1 : n = 7) (h2 : r = 2) (h3 : sum = 381) :
  let first_term := sum * (r - 1) / (r^n - 1)
  first_term = 3 := by sorry

end NUMINAMATH_CALUDE_tower_lights_problem_l921_92110


namespace NUMINAMATH_CALUDE_museum_trip_total_l921_92141

/-- The total number of people going to the museum on four buses -/
def total_people (first_bus : ℕ) : ℕ :=
  let second_bus := 2 * first_bus
  let third_bus := second_bus - 6
  let fourth_bus := first_bus + 9
  first_bus + second_bus + third_bus + fourth_bus

/-- Theorem: Given the conditions about the four buses, 
    the total number of people going to the museum is 75 -/
theorem museum_trip_total : total_people 12 = 75 := by
  sorry

end NUMINAMATH_CALUDE_museum_trip_total_l921_92141


namespace NUMINAMATH_CALUDE_root_transformation_l921_92151

/-- Given a nonzero constant k and roots a, b, c, d of the equation kx^4 - 5kx - 12 = 0,
    the polynomial with roots (b+c+d)/(ka^2), (a+c+d)/(kb^2), (a+b+d)/(kc^2), (a+b+c)/(kd^2)
    is 12k^3x^4 - 5k^3x^3 - 1 = 0 -/
theorem root_transformation (k : ℝ) (a b c d : ℝ) : k ≠ 0 →
  (k * a^4 - 5*k*a - 12 = 0) →
  (k * b^4 - 5*k*b - 12 = 0) →
  (k * c^4 - 5*k*c - 12 = 0) →
  (k * d^4 - 5*k*d - 12 = 0) →
  ∃ (x : ℝ), 12*k^3*x^4 - 5*k^3*x^3 - 1 = 0 ∧
    (x = (b+c+d)/(k*a^2) ∨ x = (a+c+d)/(k*b^2) ∨ x = (a+b+d)/(k*c^2) ∨ x = (a+b+c)/(k*d^2)) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l921_92151


namespace NUMINAMATH_CALUDE_b_arrives_first_l921_92127

theorem b_arrives_first (x y S : ℝ) (hx : x > 0) (hy : y > 0) (hS : S > 0) (hxy : x < y) :
  (S * (x + y)) / (2 * x * y) > (2 * S) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_b_arrives_first_l921_92127


namespace NUMINAMATH_CALUDE_line_perpendicular_plane_implies_planes_perpendicular_l921_92195

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the contained relation between a line and a plane
variable (contained : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perpendicularPlanes : Plane → Plane → Prop)

-- Theorem statement
theorem line_perpendicular_plane_implies_planes_perpendicular
  (m : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : contained m β) :
  perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_plane_implies_planes_perpendicular_l921_92195


namespace NUMINAMATH_CALUDE_total_distance_traveled_l921_92165

-- Define the speeds and conversion factors
def two_sail_speed : ℝ := 50
def one_sail_speed : ℝ := 25
def nautical_to_land_miles : ℝ := 1.15

-- Define the journey segments
def segment1_hours : ℝ := 2
def segment2_hours : ℝ := 3
def segment3_hours : ℝ := 1
def segment4_hours : ℝ := 2
def segment4_speed_reduction : ℝ := 0.3

-- Define the theorem
theorem total_distance_traveled :
  let segment1_distance := one_sail_speed * segment1_hours
  let segment2_distance := two_sail_speed * segment2_hours
  let segment3_distance := one_sail_speed * segment3_hours
  let segment4_distance := (one_sail_speed * (1 - segment4_speed_reduction)) * segment4_hours
  let total_nautical_miles := segment1_distance + segment2_distance + segment3_distance + segment4_distance
  let total_land_miles := total_nautical_miles * nautical_to_land_miles
  total_land_miles = 299 := by sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l921_92165


namespace NUMINAMATH_CALUDE_vector_sum_equals_l921_92126

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := 2 • a + b

theorem vector_sum_equals : c = (5, 0) := by sorry

end NUMINAMATH_CALUDE_vector_sum_equals_l921_92126


namespace NUMINAMATH_CALUDE_carrot_price_l921_92134

/-- Calculates the price of a carrot given the number of tomatoes and carrots,
    the price of a tomato, and the total revenue from selling all produce. -/
theorem carrot_price
  (num_tomatoes : ℕ)
  (num_carrots : ℕ)
  (tomato_price : ℚ)
  (total_revenue : ℚ)
  (h1 : num_tomatoes = 200)
  (h2 : num_carrots = 350)
  (h3 : tomato_price = 1)
  (h4 : total_revenue = 725) :
  (total_revenue - num_tomatoes * tomato_price) / num_carrots = 3/2 := by
  sorry

#eval (725 : ℚ) - 200 * 1
#eval ((725 : ℚ) - 200 * 1) / 350

end NUMINAMATH_CALUDE_carrot_price_l921_92134


namespace NUMINAMATH_CALUDE_minimal_shots_to_hit_triangle_l921_92162

/-- A point on the circle --/
structure Point where
  index : Nat
  h_index : index ≥ 1 ∧ index ≤ 29

/-- A shot is a pair of distinct points --/
structure Shot where
  p1 : Point
  p2 : Point
  h_distinct : p1.index ≠ p2.index

/-- A triangle on the circle --/
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point
  h_distinct : v1.index ≠ v2.index ∧ v2.index ≠ v3.index ∧ v3.index ≠ v1.index

/-- A function to determine if a shot hits a triangle --/
def hits (s : Shot) (t : Triangle) : Prop :=
  sorry -- Implementation details omitted

/-- The main theorem --/
theorem minimal_shots_to_hit_triangle :
  ∀ t : Triangle, ∃ K : Nat, K = 100 ∧
    (∀ shots : Finset Shot, shots.card = K →
      (∀ s ∈ shots, hits s t)) ∧
    (∀ K' : Nat, K' < K →
      ∃ shots : Finset Shot, shots.card = K' ∧
        ∃ s ∈ shots, ¬hits s t) :=
sorry

end NUMINAMATH_CALUDE_minimal_shots_to_hit_triangle_l921_92162


namespace NUMINAMATH_CALUDE_train_platform_length_equality_l921_92128

def train_speed : ℝ := 144  -- km/hr
def crossing_time : ℝ := 1  -- minute
def train_length : ℝ := 1200  -- meters

theorem train_platform_length_equality :
  let platform_length := train_speed * 1000 / 60 * crossing_time - train_length
  platform_length = train_length :=
by sorry

end NUMINAMATH_CALUDE_train_platform_length_equality_l921_92128


namespace NUMINAMATH_CALUDE_quadratic_real_equal_roots_l921_92191

theorem quadratic_real_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 2 * x + 5 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y + 2 * y + 5 = 0 → y = x) ↔ 
  (m = 2 - 2 * Real.sqrt 15 ∨ m = 2 + 2 * Real.sqrt 15) := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_equal_roots_l921_92191


namespace NUMINAMATH_CALUDE_cheaper_candy_price_l921_92183

/-- Proves that the price of the cheaper candy is $2 per pound -/
theorem cheaper_candy_price
  (total_weight : ℝ)
  (mixture_price : ℝ)
  (cheaper_weight : ℝ)
  (expensive_price : ℝ)
  (h1 : total_weight = 80)
  (h2 : mixture_price = 2.20)
  (h3 : cheaper_weight = 64)
  (h4 : expensive_price = 3)
  : ∃ (cheaper_price : ℝ),
    cheaper_price * cheaper_weight + expensive_price * (total_weight - cheaper_weight) =
    mixture_price * total_weight ∧ cheaper_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_cheaper_candy_price_l921_92183


namespace NUMINAMATH_CALUDE_coefficient_is_negative_seven_l921_92171

-- Define the expression
def expression (x : ℝ) : ℝ := 5 * (x - 6) + 6 * (9 - 3 * x^2 + 3 * x) - 10 * (3 * x - 2)

-- Define the coefficient of x
def coefficient_of_x (f : ℝ → ℝ) : ℝ :=
  (f 1 - f 0)

-- Theorem statement
theorem coefficient_is_negative_seven :
  coefficient_of_x expression = -7 := by sorry

end NUMINAMATH_CALUDE_coefficient_is_negative_seven_l921_92171


namespace NUMINAMATH_CALUDE_product_mod_400_l921_92118

theorem product_mod_400 : (1567 * 2150) % 400 = 50 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_400_l921_92118


namespace NUMINAMATH_CALUDE_min_balls_same_color_l921_92121

/-- Given a bag with 6 balls each of 4 different colors, the minimum number of balls
    that must be drawn to ensure two balls of the same color are drawn is 5. -/
theorem min_balls_same_color (num_colors : ℕ) (balls_per_color : ℕ) :
  num_colors = 4 →
  balls_per_color = 6 →
  5 = Nat.succ num_colors :=
by sorry

end NUMINAMATH_CALUDE_min_balls_same_color_l921_92121


namespace NUMINAMATH_CALUDE_abracadabra_anagram_count_l921_92190

/-- Represents the frequency of each letter in a word -/
structure LetterFrequency where
  a : Nat
  b : Nat
  r : Nat
  c : Nat
  d : Nat

/-- Calculates the number of anagrams for a word with given letter frequencies -/
def anagramCount (freq : LetterFrequency) : Nat :=
  Nat.factorial 11 / (Nat.factorial freq.a * Nat.factorial freq.b * Nat.factorial freq.r)

/-- The letter frequency of "ABRACADABRA" -/
def abracadabraFreq : LetterFrequency := {
  a := 5,
  b := 2,
  r := 2,
  c := 1,
  d := 1
}

theorem abracadabra_anagram_count :
  anagramCount abracadabraFreq = 83160 := by sorry

end NUMINAMATH_CALUDE_abracadabra_anagram_count_l921_92190


namespace NUMINAMATH_CALUDE_book_cost_l921_92160

theorem book_cost (book bookmark : ℝ) 
  (total_cost : book + bookmark = 2.10)
  (price_difference : book = bookmark + 2) :
  book = 2.05 := by
sorry

end NUMINAMATH_CALUDE_book_cost_l921_92160


namespace NUMINAMATH_CALUDE_penguin_giraffe_ratio_l921_92157

/-- Represents the zoo with its animal composition -/
structure Zoo where
  total_animals : ℕ
  giraffes : ℕ
  penguins : ℕ
  elephants : ℕ

/-- The conditions of the zoo -/
def zoo_conditions (z : Zoo) : Prop :=
  z.giraffes = 5 ∧
  z.penguins = (20 : ℕ) * z.total_animals / 100 ∧
  z.elephants = 2 ∧
  z.elephants = (4 : ℕ) * z.total_animals / 100

/-- The theorem stating the ratio of penguins to giraffes -/
theorem penguin_giraffe_ratio (z : Zoo) (h : zoo_conditions z) : 
  z.penguins / z.giraffes = 2 := by
  sorry

#check penguin_giraffe_ratio

end NUMINAMATH_CALUDE_penguin_giraffe_ratio_l921_92157


namespace NUMINAMATH_CALUDE_green_candy_pieces_l921_92152

theorem green_candy_pieces (total red blue : ℝ) (h1 : total = 3409.7) (h2 : red = 145.5) (h3 : blue = 785.2) :
  total - red - blue = 2479 := by
  sorry

end NUMINAMATH_CALUDE_green_candy_pieces_l921_92152


namespace NUMINAMATH_CALUDE_seating_theorem_l921_92174

/-- Number of seats in a row -/
def num_seats : ℕ := 7

/-- Number of people to be seated -/
def num_people : ℕ := 3

/-- Function to calculate the number of seating arrangements -/
def seating_arrangements (seats : ℕ) (people : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of seating arrangements -/
theorem seating_theorem :
  seating_arrangements num_seats num_people = 100 :=
sorry

end NUMINAMATH_CALUDE_seating_theorem_l921_92174


namespace NUMINAMATH_CALUDE_exists_divisible_by_3_and_19_l921_92179

theorem exists_divisible_by_3_and_19 : ∃ x : ℝ, ∃ m n : ℤ, x = 3 * m ∧ x = 19 * n := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_3_and_19_l921_92179


namespace NUMINAMATH_CALUDE_solve_linear_equation_l921_92168

theorem solve_linear_equation :
  ∃ x : ℚ, -3 * x - 12 = 6 * x + 9 → x = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l921_92168


namespace NUMINAMATH_CALUDE_successive_integers_product_l921_92120

theorem successive_integers_product (n : ℤ) : n * (n + 1) = 4160 → n = 64 := by
  sorry

end NUMINAMATH_CALUDE_successive_integers_product_l921_92120


namespace NUMINAMATH_CALUDE_class_average_l921_92148

theorem class_average (total_students : ℕ) (high_scorers : ℕ) (zero_scorers : ℕ) (high_score : ℕ) (remaining_avg : ℚ) : 
  total_students = 40 →
  high_scorers = 6 →
  zero_scorers = 9 →
  high_score = 98 →
  remaining_avg = 57 →
  (high_scorers * high_score + zero_scorers * 0 + (total_students - high_scorers - zero_scorers) * remaining_avg) / total_students = 50.325 := by
  sorry

#eval (6 * 98 + 9 * 0 + (40 - 6 - 9) * 57) / 40

end NUMINAMATH_CALUDE_class_average_l921_92148


namespace NUMINAMATH_CALUDE_max_value_z_l921_92145

/-- The maximum value of z = x - 2y subject to constraints -/
theorem max_value_z (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : 2 * x + y ≤ 2) :
  ∃ (max_z : ℝ), max_z = 1 ∧ ∀ (z : ℝ), z = x - 2 * y → z ≤ max_z :=
by sorry

end NUMINAMATH_CALUDE_max_value_z_l921_92145


namespace NUMINAMATH_CALUDE_min_value_quadratic_l921_92135

theorem min_value_quadratic (x y : ℝ) :
  x^2 + y^2 - 8*x - 6*y + 20 ≥ -5 ∧
  ∃ (a b : ℝ), a^2 + b^2 - 8*a - 6*b + 20 = -5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l921_92135


namespace NUMINAMATH_CALUDE_fraction_division_addition_l921_92164

theorem fraction_division_addition : (3 / 7 : ℚ) / 4 + 2 = 59 / 28 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_addition_l921_92164


namespace NUMINAMATH_CALUDE_middle_term_arithmetic_sequence_l921_92178

def arithmetic_sequence (a₁ a₂ a₃ a₄ a₅ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ - a₁ = d ∧ a₃ - a₂ = d ∧ a₄ - a₃ = d ∧ a₅ - a₄ = d

theorem middle_term_arithmetic_sequence :
  ∀ (a c : ℝ), arithmetic_sequence 17 a 29 c 41 → 29 = (17 + 41) / 2 :=
by sorry

end NUMINAMATH_CALUDE_middle_term_arithmetic_sequence_l921_92178


namespace NUMINAMATH_CALUDE_employee_remaining_hours_l921_92137

/-- Calculates the remaining hours for an employee who uses half of their allotted sick and vacation days --/
def remaining_hours (sick_days : ℕ) (vacation_days : ℕ) (hours_per_day : ℕ) : ℕ :=
  let remaining_sick_days := sick_days / 2
  let remaining_vacation_days := vacation_days / 2
  (remaining_sick_days + remaining_vacation_days) * hours_per_day

/-- Proves that an employee with 10 sick days and 10 vacation days, using half of each, has 80 hours left --/
theorem employee_remaining_hours :
  remaining_hours 10 10 8 = 80 := by
  sorry

end NUMINAMATH_CALUDE_employee_remaining_hours_l921_92137


namespace NUMINAMATH_CALUDE_beijing_winter_olympics_assignment_schemes_l921_92111

/-- The number of ways to assign volunteers to events -/
def assignment_schemes (n m : ℕ) : ℕ :=
  (n.choose 2) * m.factorial

/-- Theorem stating the number of assignment schemes for 5 volunteers and 4 events -/
theorem beijing_winter_olympics_assignment_schemes :
  assignment_schemes 5 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_beijing_winter_olympics_assignment_schemes_l921_92111


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l921_92100

/-- The value of p for which the left focus of the hyperbola x²/3 - y² = 1
    is on the directrix of the parabola y² = 2px -/
theorem hyperbola_parabola_intersection (p : ℝ) : p = 4 := by
  -- Define the hyperbola equation
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 3 - y^2 = 1
  -- Define the parabola equation
  let parabola := fun (x y : ℝ) ↦ y^2 = 2 * p * x
  -- Define the condition that the left focus of the hyperbola is on the directrix of the parabola
  let focus_on_directrix := ∃ (x y : ℝ), hyperbola x y ∧ parabola x y
  sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l921_92100


namespace NUMINAMATH_CALUDE_stating_valid_arrangements_count_l921_92131

/-- 
Given n players with distinct heights, this function returns the number of ways to 
arrange them such that for each player, the total number of players either to their 
left and taller or to their right and shorter is even.
-/
def validArrangements (n : ℕ) : ℕ :=
  (n / 2).factorial * ((n + 1) / 2).factorial

/-- 
Theorem stating that the number of valid arrangements for n players
is equal to ⌊n/2⌋! * ⌈n/2⌉!
-/
theorem valid_arrangements_count (n : ℕ) :
  validArrangements n = (n / 2).factorial * ((n + 1) / 2).factorial := by
  sorry

end NUMINAMATH_CALUDE_stating_valid_arrangements_count_l921_92131


namespace NUMINAMATH_CALUDE_rectangular_plot_ratio_l921_92194

/-- For a rectangular plot with given conditions, prove the ratio of area to breadth -/
theorem rectangular_plot_ratio (b l : ℝ) (h1 : b = 5) (h2 : l - b = 10) : 
  (l * b) / b = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_ratio_l921_92194


namespace NUMINAMATH_CALUDE_plane_equation_proof_l921_92196

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane given by its coefficients -/
def pointLiesOnPlane (p : Point3D) (coeff : PlaneCoefficients) : Prop :=
  coeff.A * p.x + coeff.B * p.y + coeff.C * p.z + coeff.D = 0

/-- The greatest common divisor of the absolute values of four integers is 1 -/
def gcdIsOne (a b c d : ℤ) : Prop :=
  Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Nat.gcd (Int.natAbs c) (Int.natAbs d)) = 1

theorem plane_equation_proof (p1 p2 p3 : Point3D) (coeff : PlaneCoefficients) : 
  p1 = ⟨2, -3, 1⟩ →
  p2 = ⟨6, -3, 3⟩ →
  p3 = ⟨4, -5, 2⟩ →
  coeff = ⟨2, 3, -4, 9⟩ →
  pointLiesOnPlane p1 coeff ∧
  pointLiesOnPlane p2 coeff ∧
  pointLiesOnPlane p3 coeff ∧
  coeff.A > 0 ∧
  gcdIsOne coeff.A coeff.B coeff.C coeff.D := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l921_92196


namespace NUMINAMATH_CALUDE_class_representation_ratio_l921_92188

theorem class_representation_ratio :
  ∀ (num_boys num_girls : ℕ),
  num_boys > 0 →
  num_girls > 0 →
  (num_boys : ℚ) / (num_boys + num_girls : ℚ) = 3 / 5 * (num_girls : ℚ) / (num_boys + num_girls : ℚ) →
  (num_boys : ℚ) / (num_boys + num_girls : ℚ) = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_class_representation_ratio_l921_92188


namespace NUMINAMATH_CALUDE_candy_block_pieces_l921_92117

/-- The number of candy pieces per block in Jan's candy necklace problem -/
def candy_pieces_per_block (total_necklaces : ℕ) (pieces_per_necklace : ℕ) (total_blocks : ℕ) : ℕ :=
  (total_necklaces * pieces_per_necklace) / total_blocks

/-- Theorem stating that the number of candy pieces per block is 30 -/
theorem candy_block_pieces :
  candy_pieces_per_block 9 10 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_candy_block_pieces_l921_92117


namespace NUMINAMATH_CALUDE_bus_ride_cost_l921_92124

-- Define the cost of bus and train rides
def bus_cost : ℝ := 1.75
def train_cost : ℝ := bus_cost + 6.35

-- State the theorem
theorem bus_ride_cost : 
  (train_cost = bus_cost + 6.35) → 
  (train_cost + bus_cost = 9.85) → 
  (bus_cost = 1.75) :=
by
  sorry

end NUMINAMATH_CALUDE_bus_ride_cost_l921_92124


namespace NUMINAMATH_CALUDE_total_coins_is_21_l921_92136

/-- The number of quarters in the wallet -/
def num_quarters : ℕ := 8

/-- The number of nickels in the wallet -/
def num_nickels : ℕ := 13

/-- The total number of coins in the wallet -/
def total_coins : ℕ := num_quarters + num_nickels

/-- Theorem stating that the total number of coins is 21 -/
theorem total_coins_is_21 : total_coins = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_is_21_l921_92136


namespace NUMINAMATH_CALUDE_symmetric_point_correct_l921_92163

/-- The line of symmetry -/
def line_of_symmetry (x y : ℝ) : Prop := x + 3 * y - 10 = 0

/-- The original point -/
def original_point : ℝ × ℝ := (3, 9)

/-- The symmetric point -/
def symmetric_point : ℝ × ℝ := (-1, -3)

/-- Predicate to check if a point is symmetric to another point with respect to a line -/
def is_symmetric (p1 p2 : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  line midpoint.1 midpoint.2 ∧
  (p2.2 - p1.2) * (p2.1 - p1.1) = -(1 / 3)

theorem symmetric_point_correct : 
  is_symmetric original_point symmetric_point line_of_symmetry :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_l921_92163


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_i_l921_92104

theorem imaginary_part_of_reciprocal_i : 
  Complex.im (1 / Complex.I) = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_i_l921_92104


namespace NUMINAMATH_CALUDE_guest_speaker_payment_l921_92109

theorem guest_speaker_payment (B : Nat) : 
  B < 10 → (100 * 2 + 10 * B + 5) % 13 = 0 → B = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_guest_speaker_payment_l921_92109


namespace NUMINAMATH_CALUDE_cube_difference_multiple_implies_sum_squares_multiple_of_sum_l921_92199

theorem cube_difference_multiple_implies_sum_squares_multiple_of_sum
  (a b c : ℕ+)
  (ha : a < 2017)
  (hb : b < 2017)
  (hc : c < 2017)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hca : c ≠ a)
  (hab_multiple : ∃ k : ℤ, (a ^ 3 : ℤ) - (b ^ 3 : ℤ) = k * 2017)
  (hbc_multiple : ∃ k : ℤ, (b ^ 3 : ℤ) - (c ^ 3 : ℤ) = k * 2017)
  (hca_multiple : ∃ k : ℤ, (c ^ 3 : ℤ) - (a ^ 3 : ℤ) = k * 2017) :
  ∃ m : ℕ, (a ^ 2 + b ^ 2 + c ^ 2 : ℕ) = m * (a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_cube_difference_multiple_implies_sum_squares_multiple_of_sum_l921_92199


namespace NUMINAMATH_CALUDE_circle_ratio_theorem_l921_92114

noncomputable def circle_ratio : ℝ :=
  let r1 : ℝ := Real.sqrt 2
  let r2 : ℝ := 2
  let d : ℝ := Real.sqrt 3 + 1
  let common_area : ℝ := (7 * Real.pi - 6 * (Real.sqrt 3 + 1)) / 6
  let inscribed_radius : ℝ := (Real.sqrt 2 + 1 - Real.sqrt 3) / 2
  let inscribed_area : ℝ := Real.pi * inscribed_radius ^ 2
  inscribed_area / common_area

theorem circle_ratio_theorem : circle_ratio = 
  (3 * Real.pi * (3 + Real.sqrt 2 - Real.sqrt 3 - Real.sqrt 6)) / 
  (7 * Real.pi - 6 * (Real.sqrt 3 + 1)) := by sorry

end NUMINAMATH_CALUDE_circle_ratio_theorem_l921_92114


namespace NUMINAMATH_CALUDE_min_sum_abs_min_sum_abs_achieved_l921_92129

theorem min_sum_abs (x : ℝ) : 
  |x + 3| + |x + 4| + |x + 6| + |x + 8| ≥ 12 :=
by sorry

theorem min_sum_abs_achieved : 
  ∃ x : ℝ, |x + 3| + |x + 4| + |x + 6| + |x + 8| = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_abs_min_sum_abs_achieved_l921_92129


namespace NUMINAMATH_CALUDE_min_abs_diff_solution_product_l921_92182

theorem min_abs_diff_solution_product (x y : ℤ) : 
  (20 * x + 19 * y = 2019) →
  (∀ a b : ℤ, 20 * a + 19 * b = 2019 → |x - y| ≤ |a - b|) →
  x * y = 2623 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_diff_solution_product_l921_92182


namespace NUMINAMATH_CALUDE_tylers_remaining_money_l921_92158

/-- Calculates the remaining money after Tyler's purchase of scissors and erasers. -/
theorem tylers_remaining_money 
  (initial_money : ℕ) 
  (scissor_cost : ℕ) 
  (eraser_cost : ℕ) 
  (scissor_count : ℕ) 
  (eraser_count : ℕ) 
  (h1 : initial_money = 100)
  (h2 : scissor_cost = 5)
  (h3 : eraser_cost = 4)
  (h4 : scissor_count = 8)
  (h5 : eraser_count = 10) :
  initial_money - (scissor_cost * scissor_count + eraser_cost * eraser_count) = 20 := by
  sorry

#check tylers_remaining_money

end NUMINAMATH_CALUDE_tylers_remaining_money_l921_92158


namespace NUMINAMATH_CALUDE_smallest_yellow_marbles_l921_92116

theorem smallest_yellow_marbles (total : ℕ) (blue red green yellow : ℕ) : 
  blue = total / 5 →
  red = 2 * green →
  green = 10 →
  blue + red + green + yellow = total →
  yellow ≥ 10 ∧ ∀ y : ℕ, y < 10 → ¬(
    ∃ t : ℕ, t / 5 + 2 * 10 + 10 + y = t ∧ 
    t / 5 + 2 * 10 + 10 + y = blue + red + green + y
  ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_yellow_marbles_l921_92116


namespace NUMINAMATH_CALUDE_profit_to_cost_ratio_l921_92193

theorem profit_to_cost_ratio (sale_price cost_price : ℚ) : 
  sale_price > 0 ∧ cost_price > 0 ∧ sale_price / cost_price = 6 / 2 → 
  (sale_price - cost_price) / cost_price = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_profit_to_cost_ratio_l921_92193


namespace NUMINAMATH_CALUDE_pizza_order_l921_92177

theorem pizza_order (adults children adult_slices child_slices slices_per_pizza : ℕ) 
  (h1 : adults = 2)
  (h2 : children = 6)
  (h3 : adult_slices = 3)
  (h4 : child_slices = 1)
  (h5 : slices_per_pizza = 4) :
  (adults * adult_slices + children * child_slices) / slices_per_pizza = 3 := by
  sorry


end NUMINAMATH_CALUDE_pizza_order_l921_92177


namespace NUMINAMATH_CALUDE_total_money_is_140_l921_92166

/-- Calculates the total money collected from football game tickets -/
def total_money_collected (adult_price child_price : ℚ) (total_attendees adult_attendees : ℕ) : ℚ :=
  adult_price * adult_attendees + child_price * (total_attendees - adult_attendees)

/-- Theorem stating that the total money collected is $140 -/
theorem total_money_is_140 :
  let adult_price : ℚ := 60 / 100
  let child_price : ℚ := 25 / 100
  let total_attendees : ℕ := 280
  let adult_attendees : ℕ := 200
  total_money_collected adult_price child_price total_attendees adult_attendees = 140 / 1 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_140_l921_92166


namespace NUMINAMATH_CALUDE_area_of_gray_part_l921_92144

/-- Given two overlapping rectangles, prove the area of the gray part -/
theorem area_of_gray_part (rect1_width rect1_height rect2_width rect2_height black_area : ℕ) 
  (h1 : rect1_width = 8)
  (h2 : rect1_height = 10)
  (h3 : rect2_width = 12)
  (h4 : rect2_height = 9)
  (h5 : black_area = 37) : 
  rect2_width * rect2_height - (rect1_width * rect1_height - black_area) = 65 := by
  sorry

#check area_of_gray_part

end NUMINAMATH_CALUDE_area_of_gray_part_l921_92144


namespace NUMINAMATH_CALUDE_three_fractions_inequality_l921_92150

theorem three_fractions_inequality (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_one : a + b + c = 1) :
  (a - b*c) / (a + b*c) + (b - c*a) / (b + c*a) + (c - a*b) / (c + a*b) ≤ 3/2 := by
sorry

end NUMINAMATH_CALUDE_three_fractions_inequality_l921_92150


namespace NUMINAMATH_CALUDE_point_B_coordinates_l921_92130

-- Define the points and lines
def A : ℝ × ℝ := (0, -1)
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- State the theorem
theorem point_B_coordinates :
  ∀ B : ℝ × ℝ,
  line1 B.1 B.2 →
  perpendicular ((B.2 - A.2) / (B.1 - A.1)) (-1/2) →
  B = (2, 3) := by
sorry


end NUMINAMATH_CALUDE_point_B_coordinates_l921_92130


namespace NUMINAMATH_CALUDE_yoojungs_initial_candies_l921_92189

/-- The number of candies Yoojung gave to her older sister -/
def candies_to_older_sister : ℕ := 7

/-- The number of candies Yoojung gave to her younger sister -/
def candies_to_younger_sister : ℕ := 6

/-- The number of candies Yoojung had left after giving candies to her sisters -/
def candies_left : ℕ := 15

/-- The initial number of candies Yoojung had -/
def initial_candies : ℕ := candies_to_older_sister + candies_to_younger_sister + candies_left

theorem yoojungs_initial_candies : initial_candies = 28 := by
  sorry

end NUMINAMATH_CALUDE_yoojungs_initial_candies_l921_92189


namespace NUMINAMATH_CALUDE_yardley_snowfall_l921_92186

/-- The total snowfall in Yardley throughout the day -/
def total_snowfall (early_morning late_morning afternoon evening : Real) : Real :=
  early_morning + late_morning + afternoon + evening

/-- Theorem: The total snowfall in Yardley is 1.22 inches -/
theorem yardley_snowfall :
  total_snowfall 0.12 0.24 0.5 0.36 = 1.22 := by
  sorry

end NUMINAMATH_CALUDE_yardley_snowfall_l921_92186


namespace NUMINAMATH_CALUDE_play_area_calculation_l921_92170

/-- Calculates the area of a rectangular play area given specific fencing conditions. -/
theorem play_area_calculation (total_posts : ℕ) (post_spacing : ℕ) (extra_posts_long_side : ℕ) : 
  total_posts = 24 → 
  post_spacing = 5 → 
  extra_posts_long_side = 6 → 
  ∃ (short_side_posts long_side_posts : ℕ),
    short_side_posts + extra_posts_long_side = long_side_posts ∧
    2 * short_side_posts + 2 * long_side_posts - 4 = total_posts ∧
    (short_side_posts - 1) * post_spacing * (long_side_posts - 1) * post_spacing = 675 :=
by sorry

end NUMINAMATH_CALUDE_play_area_calculation_l921_92170


namespace NUMINAMATH_CALUDE_total_score_approximation_l921_92153

/-- Represents the types of shots in a basketball game -/
inductive ShotType
  | ThreePoint
  | TwoPoint
  | FreeThrow

/-- Represents the success rate for each shot type -/
def successRate (shot : ShotType) : ℝ :=
  match shot with
  | ShotType.ThreePoint => 0.25
  | ShotType.TwoPoint => 0.50
  | ShotType.FreeThrow => 0.80

/-- Represents the point value for each shot type -/
def pointValue (shot : ShotType) : ℕ :=
  match shot with
  | ShotType.ThreePoint => 3
  | ShotType.TwoPoint => 2
  | ShotType.FreeThrow => 1

/-- The total number of shots attempted -/
def totalShots : ℕ := 40

/-- Calculates the number of attempts for each shot type, assuming equal distribution -/
def attemptsPerType : ℕ := totalShots / 3

/-- Calculates the points scored for a given shot type -/
def pointsScored (shot : ShotType) : ℝ :=
  (successRate shot) * (pointValue shot : ℝ) * (attemptsPerType : ℝ)

/-- Calculates the total points scored across all shot types -/
def totalPointsScored : ℝ :=
  pointsScored ShotType.ThreePoint + pointsScored ShotType.TwoPoint + pointsScored ShotType.FreeThrow

/-- Theorem stating that the total points scored is approximately 33 -/
theorem total_score_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |totalPointsScored - 33| < ε :=
sorry

end NUMINAMATH_CALUDE_total_score_approximation_l921_92153


namespace NUMINAMATH_CALUDE_photographer_photos_to_include_l921_92175

/-- Given a photographer with pre-selected photos and choices to provide photos,
    calculate the number of photos to include in an envelope. -/
def photos_to_include (pre_selected : ℕ) (choices : ℕ) : ℕ :=
  choices / pre_selected

/-- Theorem stating that for a photographer with 7 pre-selected photos and 56 choices,
    the number of photos to include is 8. -/
theorem photographer_photos_to_include :
  photos_to_include 7 56 = 8 := by
  sorry

end NUMINAMATH_CALUDE_photographer_photos_to_include_l921_92175


namespace NUMINAMATH_CALUDE_steak_meal_cost_l921_92102

theorem steak_meal_cost (total_initial : ℚ) (num_steaks : ℕ) 
  (burger_cost : ℚ) (ice_cream_cost : ℚ) (remaining : ℚ) :
  total_initial = 99 →
  num_steaks = 2 →
  burger_cost = 2 * 3.5 →
  ice_cream_cost = 3 * 2 →
  remaining = 38 →
  ∃ (steak_cost : ℚ), 
    total_initial - (num_steaks * steak_cost + burger_cost + ice_cream_cost) = remaining ∧
    steak_cost = 24 :=
by sorry

end NUMINAMATH_CALUDE_steak_meal_cost_l921_92102


namespace NUMINAMATH_CALUDE_fertilizer_amounts_l921_92103

def petunia_flats : ℕ := 4
def petunias_per_flat : ℕ := 8
def rose_flats : ℕ := 3
def roses_per_flat : ℕ := 6
def sunflower_flats : ℕ := 5
def sunflowers_per_flat : ℕ := 10
def orchid_flats : ℕ := 2
def orchids_per_flat : ℕ := 4
def venus_flytraps : ℕ := 2

def petunia_fertilizer_A : ℕ := 8
def rose_fertilizer_B : ℕ := 3
def sunflower_fertilizer_B : ℕ := 6
def orchid_fertilizer_A : ℕ := 4
def orchid_fertilizer_B : ℕ := 4
def venus_flytrap_fertilizer_C : ℕ := 2

theorem fertilizer_amounts :
  let total_fertilizer_A := petunia_flats * petunias_per_flat * petunia_fertilizer_A +
                            orchid_flats * orchids_per_flat * orchid_fertilizer_A
  let total_fertilizer_B := rose_flats * roses_per_flat * rose_fertilizer_B +
                            sunflower_flats * sunflowers_per_flat * sunflower_fertilizer_B +
                            orchid_flats * orchids_per_flat * orchid_fertilizer_B
  let total_fertilizer_C := venus_flytraps * venus_flytrap_fertilizer_C
  total_fertilizer_A = 288 ∧
  total_fertilizer_B = 386 ∧
  total_fertilizer_C = 4 :=
by sorry

end NUMINAMATH_CALUDE_fertilizer_amounts_l921_92103


namespace NUMINAMATH_CALUDE_max_distance_on_circle_common_chord_equation_three_common_tangents_l921_92146

-- Define the circles
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 12 = 0
def circle_C3 (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0
def circle_C4 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 8*y + 4 = 0

-- Theorem 1
theorem max_distance_on_circle :
  ∀ x₁ y₁ : ℝ, circle_C x₁ y₁ → (∀ x y : ℝ, circle_C x y → (x - 1)^2 + (y - 2*Real.sqrt 2)^2 ≤ (x₁ - 1)^2 + (y₁ - 2*Real.sqrt 2)^2) →
  (x₁ - 1)^2 + (y₁ - 2*Real.sqrt 2)^2 = 25 :=
sorry

-- Theorem 2
theorem common_chord_equation :
  ∀ x y : ℝ, (circle_C1 x y ∧ circle_C2 x y) → x - 2*y + 6 = 0 :=
sorry

-- Theorem 3
theorem three_common_tangents :
  ∃! n : ℕ, n = 3 ∧ 
  (∀ l : ℝ → ℝ → Prop, (∀ x y : ℝ, (circle_C3 x y → l x y) ∧ (circle_C4 x y → l x y)) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂)) →
  n = 3 :=
sorry

end NUMINAMATH_CALUDE_max_distance_on_circle_common_chord_equation_three_common_tangents_l921_92146


namespace NUMINAMATH_CALUDE_unique_integer_fraction_l921_92140

theorem unique_integer_fraction : ∃! n : ℕ, 
  1 ≤ n ∧ n ≤ 2014 ∧ ∃ k : ℤ, 8 * n = k * (9999 - n) := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_fraction_l921_92140


namespace NUMINAMATH_CALUDE_ginger_wears_size_8_l921_92154

def anna_size : ℕ := 2

def becky_size (anna : ℕ) : ℕ := 3 * anna

def ginger_size (becky : ℕ) : ℕ := 2 * becky - 4

theorem ginger_wears_size_8 : 
  ginger_size (becky_size anna_size) = 8 := by sorry

end NUMINAMATH_CALUDE_ginger_wears_size_8_l921_92154


namespace NUMINAMATH_CALUDE_product_equals_three_l921_92172

theorem product_equals_three (a b c d : ℚ) 
  (ha : a + 3 = 3 * a)
  (hb : b + 4 = 4 * b)
  (hc : c + 5 = 5 * c)
  (hd : d + 6 = 6 * d) : 
  a * b * c * d = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_three_l921_92172


namespace NUMINAMATH_CALUDE_positive_integer_triplets_l921_92132

theorem positive_integer_triplets :
  ∀ x y z : ℕ+,
    x ≤ y ∧ y ≤ z ∧ (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = 1 ↔
    (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 3 ∧ y = 3 ∧ z = 3) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_triplets_l921_92132


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l921_92113

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (x + 1)

theorem f_derivative_at_zero : 
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l921_92113


namespace NUMINAMATH_CALUDE_star_arrangements_l921_92122

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- The number of distinct arrangements of 12 different objects on a regular six-pointed star,
    where rotations are considered equivalent but reflections are not. -/
theorem star_arrangements : (factorial 12) / 6 = 79833600 := by
  sorry

end NUMINAMATH_CALUDE_star_arrangements_l921_92122


namespace NUMINAMATH_CALUDE_remainder_problem_l921_92198

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k < 41) 
  (h3 : k % 5 = 2) 
  (h4 : k % 6 = 5) : 
  k % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l921_92198


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l921_92156

theorem gcd_factorial_eight_and_factorial_six_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 2880 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l921_92156


namespace NUMINAMATH_CALUDE_sum_equals_14x_l921_92125

-- Define variables
variable (x y z : ℝ)

-- State the theorem
theorem sum_equals_14x (h1 : y = 3 * x) (h2 : z = 3 * y + x) : 
  x + y + z = 14 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_14x_l921_92125


namespace NUMINAMATH_CALUDE_hyperbola_minor_axis_length_l921_92119

/-- Given a hyperbola with equation x²/4 - y²/b² = 1 where b > 0,
    if the distance from the foci to the asymptote is 3,
    then the length of the minor axis is 6. -/
theorem hyperbola_minor_axis_length (b : ℝ) (h1 : b > 0) :
  (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1) →
  (∃ d : ℝ, d = 3 ∧ d = b) →
  (∃ l : ℝ, l = 6 ∧ l = 2 * b) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_minor_axis_length_l921_92119


namespace NUMINAMATH_CALUDE_pamelas_remaining_skittles_l921_92105

def initial_skittles : ℕ := 50
def skittles_given : ℕ := 7

theorem pamelas_remaining_skittles :
  initial_skittles - skittles_given = 43 := by
  sorry

end NUMINAMATH_CALUDE_pamelas_remaining_skittles_l921_92105


namespace NUMINAMATH_CALUDE_factorization_equality_l921_92180

theorem factorization_equality (x y : ℝ) : 3 * x^2 + 6 * x * y + 3 * y^2 = 3 * (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l921_92180


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l921_92115

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- The total number of possible triangles formed by choosing 3 vertices from a decagon -/
def total_triangles : ℕ := Nat.choose n k

/-- The number of triangles with at least one side being a side of the decagon -/
def favorable_triangles : ℕ := 70

/-- The probability of a triangle having at least one side that is a side of the decagon -/
def probability : ℚ := favorable_triangles / total_triangles

theorem decagon_triangle_probability :
  probability = 7 / 12 := by sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l921_92115


namespace NUMINAMATH_CALUDE_bicycle_sale_loss_percentage_l921_92173

theorem bicycle_sale_loss_percentage
  (profit_A_to_B : ℝ)
  (profit_A_to_C : ℝ)
  (h1 : profit_A_to_B = 0.30)
  (h2 : profit_A_to_C = 0.040000000000000036) :
  ∃ (loss_B_to_C : ℝ), loss_B_to_C = 0.20 ∧ 
    (1 + profit_A_to_C) = (1 + profit_A_to_B) * (1 - loss_B_to_C) := by
  sorry

end NUMINAMATH_CALUDE_bicycle_sale_loss_percentage_l921_92173


namespace NUMINAMATH_CALUDE_correct_divisor_l921_92101

theorem correct_divisor (X D : ℕ) (h1 : X % D = 0) (h2 : X % 12 = 0) (h3 : X / 12 = 56) (h4 : X / D = 32) : D = 21 := by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_l921_92101


namespace NUMINAMATH_CALUDE_iphone_price_calculation_l921_92138

theorem iphone_price_calculation (P : ℝ) : 
  (P * (1 - 0.1) * (1 - 0.2) = 720) → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_iphone_price_calculation_l921_92138


namespace NUMINAMATH_CALUDE_cube_root_3a_5b_square_root_4x_y_l921_92112

-- Part 1
theorem cube_root_3a_5b (a b : ℝ) (h : b = 4 * Real.sqrt (3 * a - 2) + 2 * Real.sqrt (2 - 3 * a) + 5) :
  (3 * a + 5 * b) ^ (1/3 : ℝ) = 3 := by sorry

-- Part 2
theorem square_root_4x_y (x y : ℝ) (h : (x - 3)^2 + Real.sqrt (y - 4) = 0) :
  (4 * x + y) ^ (1/2 : ℝ) = 4 ∨ (4 * x + y) ^ (1/2 : ℝ) = -4 := by sorry

end NUMINAMATH_CALUDE_cube_root_3a_5b_square_root_4x_y_l921_92112


namespace NUMINAMATH_CALUDE_remainders_of_1493827_l921_92139

theorem remainders_of_1493827 : 
  (1493827 % 4 = 3) ∧ (1493827 % 3 = 1) := by
  sorry

end NUMINAMATH_CALUDE_remainders_of_1493827_l921_92139


namespace NUMINAMATH_CALUDE_sum_first_six_terms_eq_54_l921_92167

/-- An arithmetic sequence with given 3rd, 4th, and 5th terms -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  a 3 = 7 ∧ a 4 = 11 ∧ a 5 = 15 ∧ ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The sum of the first six terms of the sequence -/
def SumFirstSixTerms (a : ℕ → ℤ) : ℤ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

/-- Theorem stating that the sum of the first six terms is 54 -/
theorem sum_first_six_terms_eq_54 (a : ℕ → ℤ) (h : ArithmeticSequence a) :
  SumFirstSixTerms a = 54 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_terms_eq_54_l921_92167


namespace NUMINAMATH_CALUDE_tony_purchase_cost_l921_92169

/-- Calculates the total cost of Tony's purchases given the specified conditions --/
def total_cost (lego_price : ℝ) (sword_price_eur : ℝ) (dough_price_gbp : ℝ)
                (day1_discount : ℝ) (day2_discount : ℝ) (sales_tax : ℝ)
                (eur_to_usd_day1 : ℝ) (gbp_to_usd_day1 : ℝ)
                (eur_to_usd_day2 : ℝ) (gbp_to_usd_day2 : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the total cost is $1560.83 given the problem conditions --/
theorem tony_purchase_cost :
  let lego_price := 250
  let sword_price_eur := 100
  let dough_price_gbp := 30
  let day1_discount := 0.2
  let day2_discount := 0.1
  let sales_tax := 0.05
  let eur_to_usd_day1 := 1 / 0.85
  let gbp_to_usd_day1 := 1 / 0.75
  let eur_to_usd_day2 := 1 / 0.84
  let gbp_to_usd_day2 := 1 / 0.74
  total_cost lego_price sword_price_eur dough_price_gbp
             day1_discount day2_discount sales_tax
             eur_to_usd_day1 gbp_to_usd_day1
             eur_to_usd_day2 gbp_to_usd_day2 = 1560.83 :=
by
  sorry

end NUMINAMATH_CALUDE_tony_purchase_cost_l921_92169


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l921_92143

theorem floor_plus_self_unique_solution :
  ∃! s : ℝ, ⌊s⌋ + s = 22.7 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l921_92143


namespace NUMINAMATH_CALUDE_wage_decrease_increase_l921_92161

theorem wage_decrease_increase (initial_wage : ℝ) :
  let decreased_wage := initial_wage * (1 - 0.5)
  let final_wage := decreased_wage * (1 + 0.5)
  final_wage = initial_wage * 0.75 ∧ (initial_wage - final_wage) / initial_wage = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_wage_decrease_increase_l921_92161


namespace NUMINAMATH_CALUDE_mean_median_difference_zero_l921_92184

/-- Represents the score distribution in a classroom --/
structure ScoreDistribution where
  score60 : ℝ
  score75 : ℝ
  score85 : ℝ
  score90 : ℝ
  score95 : ℝ
  sum_to_one : score60 + score75 + score85 + score90 + score95 = 1

/-- Calculates the mean score given a score distribution --/
def mean_score (d : ScoreDistribution) : ℝ :=
  60 * d.score60 + 75 * d.score75 + 85 * d.score85 + 90 * d.score90 + 95 * d.score95

/-- Calculates the median score given a score distribution --/
def median_score (d : ScoreDistribution) : ℝ := 85

/-- The main theorem stating that the difference between mean and median is zero --/
theorem mean_median_difference_zero (d : ScoreDistribution) :
  d.score60 = 0.05 →
  d.score75 = 0.20 →
  d.score85 = 0.30 →
  d.score90 = 0.25 →
  mean_score d - median_score d = 0 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_zero_l921_92184


namespace NUMINAMATH_CALUDE_particle_speeds_l921_92197

-- Define the distance between points A and B in centimeters
def distance : ℝ := 301

-- Define the time when m2 starts moving after m1 leaves A
def start_time : ℝ := 11

-- Define the times of the two meetings after m2 starts moving
def first_meeting : ℝ := 10
def second_meeting : ℝ := 45

-- Define the speeds of particles m1 and m2
def speed_m1 : ℝ := 11
def speed_m2 : ℝ := 7

-- Theorem statement
theorem particle_speeds :
  -- Condition: At the first meeting, the total distance covered equals the initial distance
  (distance - start_time * speed_m1 = first_meeting * (speed_m1 + speed_m2)) ∧
  -- Condition: The relative movement between the two meetings
  (2 * first_meeting * speed_m2 = (second_meeting - first_meeting) * (speed_m1 - speed_m2)) →
  -- Conclusion: The speeds are correct
  speed_m1 = 11 ∧ speed_m2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_particle_speeds_l921_92197


namespace NUMINAMATH_CALUDE_xiaoxia_exceeds_xiaoming_l921_92181

theorem xiaoxia_exceeds_xiaoming (n : ℕ) : 
  let xiaoxia_initial : ℤ := 52
  let xiaoming_initial : ℤ := 70
  let xiaoxia_monthly : ℤ := 15
  let xiaoming_monthly : ℤ := 12
  let xiaoxia_savings : ℤ := xiaoxia_initial + xiaoxia_monthly * n
  let xiaoming_savings : ℤ := xiaoming_initial + xiaoming_monthly * n
  xiaoxia_savings > xiaoming_savings ↔ 52 + 15 * n > 70 + 12 * n :=
by sorry

end NUMINAMATH_CALUDE_xiaoxia_exceeds_xiaoming_l921_92181


namespace NUMINAMATH_CALUDE_same_last_five_digits_l921_92142

theorem same_last_five_digits (N : ℕ) : N = 3125 ↔ 
  (N > 0) ∧ 
  (∃ (a b c d e : ℕ), 
    a ≠ 0 ∧ 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
    N % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    (N^2) % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e) ∧
  (∀ M : ℕ, M < N → 
    (M > 0) → 
    (∀ (a b c d e : ℕ), 
      a ≠ 0 → 
      a < 10 → b < 10 → c < 10 → d < 10 → e < 10 →
      M % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e →
      (M^2) % 100000 ≠ a * 10000 + b * 1000 + c * 100 + d * 10 + e)) :=
by sorry

end NUMINAMATH_CALUDE_same_last_five_digits_l921_92142


namespace NUMINAMATH_CALUDE_first_supplier_cars_l921_92187

theorem first_supplier_cars (total_production : ℕ) 
  (second_supplier_extra : ℕ) (fourth_fifth_supplier : ℕ) : 
  total_production = 5650000 →
  second_supplier_extra = 500000 →
  fourth_fifth_supplier = 325000 →
  ∃ (first_supplier : ℕ),
    first_supplier + 
    (first_supplier + second_supplier_extra) + 
    (first_supplier + (first_supplier + second_supplier_extra)) + 
    (2 * fourth_fifth_supplier) = total_production ∧
    first_supplier = 1000000 :=
by sorry

end NUMINAMATH_CALUDE_first_supplier_cars_l921_92187


namespace NUMINAMATH_CALUDE_rectangle_area_l921_92106

/-- Given three similar rectangles where ABCD is the largest, prove its area --/
theorem rectangle_area (width height : ℝ) (h1 : width = 15) (h2 : height = width * Real.sqrt 6) :
  width * height = 75 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l921_92106


namespace NUMINAMATH_CALUDE_specific_pyramid_volume_l921_92147

/-- A triangular pyramid with mutually perpendicular lateral edges -/
structure TriangularPyramid where
  /-- The area of the first lateral face -/
  area1 : ℝ
  /-- The area of the second lateral face -/
  area2 : ℝ
  /-- The area of the third lateral face -/
  area3 : ℝ
  /-- The lateral edges are mutually perpendicular -/
  perpendicular : True

/-- The volume of a triangular pyramid -/
def volume (p : TriangularPyramid) : ℝ := sorry

/-- Theorem: The volume of the specific triangular pyramid is 2 cm³ -/
theorem specific_pyramid_volume :
  let p : TriangularPyramid := {
    area1 := 1.5,
    area2 := 2,
    area3 := 6,
    perpendicular := trivial
  }
  volume p = 2 := by sorry

end NUMINAMATH_CALUDE_specific_pyramid_volume_l921_92147


namespace NUMINAMATH_CALUDE_square_inequality_l921_92133

theorem square_inequality {a b : ℝ} (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l921_92133


namespace NUMINAMATH_CALUDE_pi_sixth_to_degrees_l921_92149

theorem pi_sixth_to_degrees : 
  (π / 6 : Real) * (180 / π) = 30 := by sorry

end NUMINAMATH_CALUDE_pi_sixth_to_degrees_l921_92149


namespace NUMINAMATH_CALUDE_carnival_prize_percentage_carnival_prize_percentage_proof_l921_92155

theorem carnival_prize_percentage (total_minnows : ℕ) (minnows_per_prize : ℕ) 
  (total_players : ℕ) (leftover_minnows : ℕ) : ℕ → Prop :=
  λ percentage_winners =>
    total_minnows = 600 ∧
    minnows_per_prize = 3 ∧
    total_players = 800 ∧
    leftover_minnows = 240 →
    percentage_winners = 15 ∧
    (total_minnows - leftover_minnows) / minnows_per_prize * 100 / total_players = percentage_winners

-- Proof
theorem carnival_prize_percentage_proof : 
  ∃ (percentage_winners : ℕ), carnival_prize_percentage 600 3 800 240 percentage_winners :=
by
  sorry

end NUMINAMATH_CALUDE_carnival_prize_percentage_carnival_prize_percentage_proof_l921_92155


namespace NUMINAMATH_CALUDE_gcd_8369_4087_2159_l921_92108

theorem gcd_8369_4087_2159 : Nat.gcd 8369 (Nat.gcd 4087 2159) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8369_4087_2159_l921_92108


namespace NUMINAMATH_CALUDE_final_marble_count_l921_92107

def initial_marbles : ℝ := 87.0
def received_marbles : ℝ := 8.0

theorem final_marble_count :
  initial_marbles + received_marbles = 95.0 := by sorry

end NUMINAMATH_CALUDE_final_marble_count_l921_92107


namespace NUMINAMATH_CALUDE_cubic_root_sum_l921_92123

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 10*a^2 + 16*a - 2 = 0 →
  b^3 - 10*b^2 + 16*b - 2 = 0 →
  c^3 - 10*c^2 + 16*c - 2 = 0 →
  (a / (b*c + 2)) + (b / (a*c + 2)) + (c / (a*b + 2)) = 4 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l921_92123


namespace NUMINAMATH_CALUDE_work_completion_time_l921_92159

theorem work_completion_time (work : ℝ) (a b : ℝ) 
  (h1 : a + b = work / 6)  -- A and B together complete work in 6 days
  (h2 : a = work / 14)     -- A alone completes work in 14 days
  : b = work / 10.5        -- B alone completes work in 10.5 days
:= by sorry

end NUMINAMATH_CALUDE_work_completion_time_l921_92159
