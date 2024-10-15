import Mathlib

namespace NUMINAMATH_CALUDE_train_crossing_time_l258_25842

/-- Calculates the time for a train to cross a platform -/
theorem train_crossing_time
  (train_speed : Real)
  (man_crossing_time : Real)
  (platform_length : Real)
  (h1 : train_speed = 72 * (1000 / 3600)) -- 72 kmph converted to m/s
  (h2 : man_crossing_time = 20)
  (h3 : platform_length = 200) :
  let train_length := train_speed * man_crossing_time
  let total_length := train_length + platform_length
  total_length / train_speed = 30 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l258_25842


namespace NUMINAMATH_CALUDE_camila_garden_walkway_area_camila_garden_walkway_area_proof_l258_25836

/-- The total area of walkways in Camila's garden -/
theorem camila_garden_walkway_area : ℕ :=
  let num_rows : ℕ := 4
  let num_cols : ℕ := 3
  let bed_width : ℕ := 8
  let bed_height : ℕ := 3
  let walkway_width : ℕ := 2
  let total_width : ℕ := num_cols * bed_width + (num_cols + 1) * walkway_width
  let total_height : ℕ := num_rows * bed_height + (num_rows + 1) * walkway_width
  let total_area : ℕ := total_width * total_height
  let total_bed_area : ℕ := num_rows * num_cols * bed_width * bed_height
  let walkway_area : ℕ := total_area - total_bed_area
  416

theorem camila_garden_walkway_area_proof : camila_garden_walkway_area = 416 := by
  sorry

end NUMINAMATH_CALUDE_camila_garden_walkway_area_camila_garden_walkway_area_proof_l258_25836


namespace NUMINAMATH_CALUDE_x_plus_y_range_l258_25872

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the conditions
def conditions (x y : ℝ) : Prop :=
  y = 3 * (floor x) + 4 ∧
  y = 4 * (floor (x - 3)) + 7 ∧
  x ≠ ↑(floor x)

-- Theorem statement
theorem x_plus_y_range (x y : ℝ) :
  conditions x y → 40 < x + y ∧ x + y < 41 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_range_l258_25872


namespace NUMINAMATH_CALUDE_negation_of_proposition_l258_25833

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l258_25833


namespace NUMINAMATH_CALUDE_max_sections_school_l258_25850

theorem max_sections_school (num_boys : ℕ) (num_girls : ℕ) (min_boys_per_section : ℕ) (min_girls_per_section : ℕ) 
  (h1 : num_boys = 2016) 
  (h2 : num_girls = 1284) 
  (h3 : min_boys_per_section = 80) 
  (h4 : min_girls_per_section = 60) : 
  (num_boys / min_boys_per_section + num_girls / min_girls_per_section : ℕ) = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sections_school_l258_25850


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l258_25818

def a (n : ℕ) : ℤ := 2^n - (-1)^n

theorem arithmetic_sequence_properties :
  (∃ (n₁ n₂ n₃ : ℕ), n₁ < n₂ ∧ n₂ < n₃ ∧ 
    n₂ = n₁ + 1 ∧ n₃ = n₂ + 1 ∧
    2 * a n₂ = a n₁ + a n₃ ∧ n₁ = 2) ∧
  (∀ n₂ n₃ : ℕ, 1 < n₂ ∧ n₂ < n₃ ∧ 
    2 * a n₂ = a 1 + a n₃ → n₃ - n₂ = 1) ∧
  (∀ t : ℕ, t > 3 → 
    ¬∃ (s : ℕ → ℕ), Monotone s ∧ 
      (∀ i j : Fin t, i < j → s i < s j) ∧
      (∀ i : Fin (t - 1), 
        2 * a (s (i + 1)) = a (s i) + a (s (i + 2)))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l258_25818


namespace NUMINAMATH_CALUDE_engineering_collections_l258_25834

/-- Represents the count of each letter in "ENGINEERING" -/
structure LetterCount where
  e : Nat -- vowel
  n : Nat -- consonant
  g : Nat -- consonant
  r : Nat -- consonant
  i : Nat -- consonant

/-- Represents a collection of letters -/
structure LetterCollection where
  vowels : Nat
  consonants : Nat

/-- Checks if a letter collection is valid -/
def isValidCollection (lc : LetterCollection) : Prop :=
  lc.vowels = 3 ∧ lc.consonants = 3

/-- Counts the number of distinct letter collections -/
noncomputable def countDistinctCollections (word : LetterCount) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem engineering_collections (word : LetterCount) 
  (h1 : word.e = 5) -- number of E's
  (h2 : word.n = 2) -- number of N's
  (h3 : word.g = 3) -- number of G's
  (h4 : word.r = 1) -- number of R's
  (h5 : word.i = 1) -- number of I's
  : countDistinctCollections word = 13 := by sorry

end NUMINAMATH_CALUDE_engineering_collections_l258_25834


namespace NUMINAMATH_CALUDE_smallest_angle_at_vertices_l258_25807

/-- A cube in 3D space -/
structure Cube where
  side : ℝ
  center : ℝ × ℝ × ℝ

/-- A point in 3D space -/
def Point3D := ℝ × ℝ × ℝ

/-- The angle at which a point sees the space diagonal of a cube -/
def angle_at_point (c : Cube) (p : Point3D) : ℝ := sorry

/-- The vertices of a cube -/
def cube_vertices (c : Cube) : Set Point3D := sorry

/-- The surface of a cube -/
def cube_surface (c : Cube) : Set Point3D := sorry

/-- Theorem: The vertices of a cube are the only points on its surface where 
    the space diagonal is seen at a 90-degree angle, which is the smallest possible angle -/
theorem smallest_angle_at_vertices (c : Cube) : 
  ∀ p ∈ cube_surface c, 
    angle_at_point c p = Real.pi / 2 ↔ p ∈ cube_vertices c :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_at_vertices_l258_25807


namespace NUMINAMATH_CALUDE_division_remainder_l258_25848

theorem division_remainder : ∃ q : ℤ, 1346584 = 137 * q + 5 ∧ 0 ≤ 5 ∧ 5 < 137 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l258_25848


namespace NUMINAMATH_CALUDE_iains_pennies_l258_25826

/-- The number of pennies Iain had initially -/
def initial_pennies : ℕ := 200

/-- The number of old pennies removed -/
def old_pennies : ℕ := 30

/-- The percentage of remaining pennies kept after throwing out -/
def kept_percentage : ℚ := 80 / 100

/-- The number of pennies left after removing old pennies and throwing out some -/
def remaining_pennies : ℕ := 136

theorem iains_pennies :
  (kept_percentage * (initial_pennies - old_pennies : ℚ)).floor = remaining_pennies :=
sorry

end NUMINAMATH_CALUDE_iains_pennies_l258_25826


namespace NUMINAMATH_CALUDE_red_exhausted_first_l258_25887

/-- Represents the number of marbles of each color in the bag -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the probability that red marbles are the first to be exhausted -/
def probability_red_exhausted (bag : MarbleBag) : ℚ :=
  sorry

/-- The theorem stating the probability of red marbles being exhausted first -/
theorem red_exhausted_first (bag : MarbleBag) 
  (h1 : bag.red = 3) 
  (h2 : bag.blue = 5) 
  (h3 : bag.green = 7) : 
  probability_red_exhausted bag = 21 / 40 := by
  sorry

end NUMINAMATH_CALUDE_red_exhausted_first_l258_25887


namespace NUMINAMATH_CALUDE_school_referendum_l258_25867

theorem school_referendum (U A B : Finset Nat) (h1 : Finset.card U = 250)
  (h2 : Finset.card A = 190) (h3 : Finset.card B = 150)
  (h4 : Finset.card (U \ (A ∪ B)) = 40) :
  Finset.card (A ∩ B) = 130 := by
  sorry

end NUMINAMATH_CALUDE_school_referendum_l258_25867


namespace NUMINAMATH_CALUDE_banana_arrangements_l258_25863

def word := "BANANA"
def total_letters : ℕ := 6
def freq_B : ℕ := 1
def freq_A : ℕ := 3
def freq_N : ℕ := 2

theorem banana_arrangements : 
  (Nat.factorial total_letters) / 
  (Nat.factorial freq_B * Nat.factorial freq_A * Nat.factorial freq_N) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l258_25863


namespace NUMINAMATH_CALUDE_integer_2020_in_column_F_l258_25892

/-- Represents the columns in the arrangement --/
inductive Column
  | A | B | C | D | E | F | G

/-- Defines the arrangement of integers in columns --/
def arrangement (n : ℕ) : Column :=
  match (n - 11) % 14 with
  | 0 => Column.A
  | 1 => Column.B
  | 2 => Column.C
  | 3 => Column.D
  | 4 => Column.E
  | 5 => Column.F
  | 6 => Column.G
  | 7 => Column.G
  | 8 => Column.F
  | 9 => Column.E
  | 10 => Column.D
  | 11 => Column.C
  | 12 => Column.B
  | _ => Column.A

/-- Theorem: The integer 2020 is in column F --/
theorem integer_2020_in_column_F : arrangement 2020 = Column.F := by
  sorry

end NUMINAMATH_CALUDE_integer_2020_in_column_F_l258_25892


namespace NUMINAMATH_CALUDE_expression_simplification_value_at_three_value_at_four_l258_25885

theorem expression_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  (1 - 1 / (x - 1)) / ((x^2 - 4) / (x^2 - 2*x + 1)) = (x - 1) / (x + 2) := by
  sorry

theorem value_at_three :
  (1 - 1 / (3 - 1)) / ((3^2 - 4) / (3^2 - 2*3 + 1)) = 2 / 5 := by
  sorry

theorem value_at_four :
  (1 - 1 / (4 - 1)) / ((4^2 - 4) / (4^2 - 2*4 + 1)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_value_at_three_value_at_four_l258_25885


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_product_l258_25815

/-- Represents a conic section (ellipse or hyperbola) -/
structure Conic where
  center : ℝ × ℝ
  foci : ℝ × ℝ
  eccentricity : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem ellipse_hyperbola_eccentricity_product (C₁ C₂ : Conic) (P : Point) :
  C₁.center = (0, 0) →
  C₂.center = (0, 0) →
  C₁.foci.1 < 0 →
  C₁.foci.2 > 0 →
  C₂.foci = C₁.foci →
  P.x > 0 →
  P.y > 0 →
  (P.x - C₁.foci.1)^2 + P.y^2 = (P.x - C₁.foci.2)^2 + P.y^2 →
  C₁.eccentricity * C₂.eccentricity > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_product_l258_25815


namespace NUMINAMATH_CALUDE_bakery_batches_l258_25824

/-- The number of baguettes in each batch -/
def baguettes_per_batch : ℕ := 48

/-- The number of baguettes sold after each batch -/
def baguettes_sold : List ℕ := [37, 52, 49]

/-- The number of baguettes left unsold -/
def baguettes_left : ℕ := 6

/-- The number of batches of baguettes the bakery makes a day -/
def num_batches : ℕ := 3

theorem bakery_batches :
  (baguettes_per_batch * num_batches) = (baguettes_sold.sum + baguettes_left) :=
by sorry

end NUMINAMATH_CALUDE_bakery_batches_l258_25824


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l258_25814

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_sum : a 6 + a 8 = 4) : 
  a 8 * (a 4 + 2 * a 6 + a 8) = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l258_25814


namespace NUMINAMATH_CALUDE_hike_up_time_l258_25884

/-- Proves that the time taken to hike up a hill is 1.8 hours given specific conditions -/
theorem hike_up_time (up_speed down_speed total_time : ℝ) 
  (h1 : up_speed = 4)
  (h2 : down_speed = 6)
  (h3 : total_time = 3) : 
  ∃ (t : ℝ), t * up_speed = (total_time - t) * down_speed ∧ t = 1.8 := by
  sorry

#check hike_up_time

end NUMINAMATH_CALUDE_hike_up_time_l258_25884


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l258_25827

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ = 31 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l258_25827


namespace NUMINAMATH_CALUDE_ratio_x_sqrt_w_l258_25812

theorem ratio_x_sqrt_w (x y z w v : ℝ) 
  (hx : x = 1.20 * y)
  (hy : y = 0.30 * z)
  (hz : z = 1.35 * w)
  (hw : w = v^2)
  (hv : v = 0.50 * x) :
  x / Real.sqrt w = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_sqrt_w_l258_25812


namespace NUMINAMATH_CALUDE_negative_squares_inequality_l258_25881

theorem negative_squares_inequality (x b a : ℝ) 
  (h1 : x < b) (h2 : b < a) (h3 : a < 0) : x^2 > b*x ∧ b*x > b^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_squares_inequality_l258_25881


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l258_25831

/-- 
Given the equation 2x(x+5) = 10, this theorem states that when converted to 
general form ax² + bx + c = 0, the coefficients a, b, and c are 2, 10, and -10 respectively.
-/
theorem quadratic_equation_coefficients : 
  ∃ (a b c : ℝ), (∀ x, 2*x*(x+5) = 10 ↔ a*x^2 + b*x + c = 0) ∧ a = 2 ∧ b = 10 ∧ c = -10 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l258_25831


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l258_25880

/-- Given a trapezoid ABCD, proves that if the ratio of the areas of triangles ABC and ADC is 5:2,
    and the sum of AB and CD is 280, then AB equals 200. -/
theorem trapezoid_segment_length (A B C D : Point) (h : ℝ) :
  let triangle_ABC := (1/2) * AB * h
  let triangle_ADC := (1/2) * CD * h
  triangle_ABC / triangle_ADC = 5/2 →
  AB + CD = 280 →
  AB = 200 :=
by
  sorry


end NUMINAMATH_CALUDE_trapezoid_segment_length_l258_25880


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l258_25886

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 7) :
  a / c = 105 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l258_25886


namespace NUMINAMATH_CALUDE_square_root_equation_l258_25828

theorem square_root_equation (x : ℝ) : 
  (Real.sqrt x / Real.sqrt 0.64) + (Real.sqrt 1.44 / Real.sqrt 0.49) = 3.0892857142857144 → x = 1.21 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l258_25828


namespace NUMINAMATH_CALUDE_triangle_sine_theorem_l258_25898

theorem triangle_sine_theorem (D E F : ℝ) (area : ℝ) (geo_mean : ℝ) :
  area = 81 →
  geo_mean = 15 →
  geo_mean^2 = D * F →
  area = 1/2 * D * F * Real.sin E →
  Real.sin E = 18/25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_theorem_l258_25898


namespace NUMINAMATH_CALUDE_cupcake_ratio_l258_25854

/-- Proves that the ratio of gluten-free cupcakes to total cupcakes is 3/20 given the specified conditions --/
theorem cupcake_ratio : 
  ∀ (total vegan non_vegan gluten_free : ℕ),
    total = 80 →
    vegan = 24 →
    non_vegan = 28 →
    gluten_free = vegan / 2 →
    (gluten_free : ℚ) / total = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_ratio_l258_25854


namespace NUMINAMATH_CALUDE_quadratic_inequalities_condition_l258_25802

theorem quadratic_inequalities_condition (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (∀ x, a₁ * x^2 + b₁ * x + c₁ > 0 ↔ a₂ * x^2 + b₂ * x + c₂ > 0) ↔
    (a₁ / a₂ = b₁ / b₂ ∧ b₁ / b₂ = c₁ / c₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_condition_l258_25802


namespace NUMINAMATH_CALUDE_cube_surface_area_from_volume_l258_25860

theorem cube_surface_area_from_volume (volume : ℝ) (side_length : ℝ) (surface_area : ℝ) : 
  volume = 343 →
  volume = side_length ^ 3 →
  surface_area = 6 * side_length ^ 2 →
  surface_area = 294 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_volume_l258_25860


namespace NUMINAMATH_CALUDE_triangle_properties_l258_25803

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  (1/2 * b * c * Real.sin A = 3 * Real.sin A) →
  (a + b + c = 4 * (Real.sqrt 2 + 1)) →
  (Real.sin B + Real.sin C = Real.sqrt 2 * Real.sin A) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a = 4 ∧ 
   Real.cos A = 1/3 ∧
   Real.cos (2*A - π/3) = (4*Real.sqrt 6 - 7) / 18) := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l258_25803


namespace NUMINAMATH_CALUDE_problem_solution_l258_25804

noncomputable section

def f (x : ℝ) : ℝ := Real.log ((2 / (x + 1)) - 1) / Real.log 10

def g (a x : ℝ) : ℝ := Real.sqrt (1 - a^2 - 2*a*x - x^2)

def A : Set ℝ := {x : ℝ | (2 / (x + 1)) - 1 > 0}

def B (a : ℝ) : Set ℝ := {x : ℝ | 1 - a^2 - 2*a*x - x^2 ≥ 0}

theorem problem_solution (a : ℝ) :
  (f (1/2013) + f (-1/2013) = 0) ∧
  (∀ a, a ≥ 2 → A ∩ B a = ∅) ∧
  (∃ a, a < 2 ∧ A ∩ B a = ∅) :=
sorry

end

end NUMINAMATH_CALUDE_problem_solution_l258_25804


namespace NUMINAMATH_CALUDE_specific_trip_mpg_l258_25822

/-- Represents a car trip with odometer readings and fuel consumption --/
structure CarTrip where
  initial_odometer : ℕ
  initial_fuel : ℕ
  first_refill : ℕ
  second_refill_odometer : ℕ
  second_refill_amount : ℕ
  final_odometer : ℕ
  final_refill : ℕ

/-- Calculates the average miles per gallon for a car trip --/
def averageMPG (trip : CarTrip) : ℚ :=
  let total_distance := trip.final_odometer - trip.initial_odometer
  let total_fuel := trip.initial_fuel + trip.first_refill + trip.second_refill_amount + trip.final_refill
  (total_distance : ℚ) / total_fuel

/-- The specific car trip from the problem --/
def specificTrip : CarTrip := {
  initial_odometer := 58000
  initial_fuel := 2
  first_refill := 8
  second_refill_odometer := 58400
  second_refill_amount := 15
  final_odometer := 59000
  final_refill := 25
}

/-- Theorem stating that the average MPG for the specific trip is 20.0 --/
theorem specific_trip_mpg : averageMPG specificTrip = 20 := by
  sorry

end NUMINAMATH_CALUDE_specific_trip_mpg_l258_25822


namespace NUMINAMATH_CALUDE_f_equiv_g_l258_25890

/-- Function f defined as f(x) = x^2 - 2x - 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

/-- Function g defined as g(t) = t^2 - 2t + 1 -/
def g (t : ℝ) : ℝ := t^2 - 2*t + 1

/-- Theorem stating that f and g are equivalent functions -/
theorem f_equiv_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_f_equiv_g_l258_25890


namespace NUMINAMATH_CALUDE_shelly_money_proof_l258_25845

/-- Calculates the total amount of money Shelly has given the number of $10 and $5 bills -/
def total_money (ten_dollar_bills : ℕ) (five_dollar_bills : ℕ) : ℕ :=
  10 * ten_dollar_bills + 5 * five_dollar_bills

/-- Proves that Shelly has $390 in total -/
theorem shelly_money_proof :
  let ten_dollar_bills : ℕ := 30
  let five_dollar_bills : ℕ := ten_dollar_bills - 12
  total_money ten_dollar_bills five_dollar_bills = 390 := by
sorry

end NUMINAMATH_CALUDE_shelly_money_proof_l258_25845


namespace NUMINAMATH_CALUDE_remainder_theorem_l258_25820

theorem remainder_theorem (x : ℤ) : x % 66 = 14 → x % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l258_25820


namespace NUMINAMATH_CALUDE_q_necessary_not_sufficient_for_p_l258_25853

theorem q_necessary_not_sufficient_for_p :
  (∀ x : ℝ, |x| < 1 → x^2 + x - 6 < 0) ∧
  (∃ x : ℝ, x^2 + x - 6 < 0 ∧ |x| ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_q_necessary_not_sufficient_for_p_l258_25853


namespace NUMINAMATH_CALUDE_square_difference_l258_25875

theorem square_difference (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : a^2 - b^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l258_25875


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l258_25840

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 12 < 0} = Set.Ioo (-4) 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l258_25840


namespace NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_special_case_l258_25855

/-- Given two real numbers a and b, we define A and B as follows: -/
def A (a b : ℝ) : ℝ := 3 * b^2 - 2 * a^2 + 5 * a * b

def B (a b : ℝ) : ℝ := 4 * a * b + 2 * b^2 - a^2

/-- Theorem stating that 2A - 3B simplifies to -a² - 2ab for any real a and b -/
theorem simplify_2A_minus_3B (a b : ℝ) : 2 * A a b - 3 * B a b = -a^2 - 2*a*b := by
  sorry

/-- Theorem stating that when a = -1 and b = 2, the value of 2A - 3B is 3 -/
theorem value_2A_minus_3B_special_case : 2 * A (-1) 2 - 3 * B (-1) 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_special_case_l258_25855


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_l258_25852

theorem cubic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 * b + b^3 * c + c^3 * a ≥ a * b * c * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_l258_25852


namespace NUMINAMATH_CALUDE_probability_all_female_committee_l258_25830

def total_group_size : ℕ := 8
def num_females : ℕ := 5
def num_males : ℕ := 3
def committee_size : ℕ := 3

theorem probability_all_female_committee :
  (Nat.choose num_females committee_size : ℚ) / (Nat.choose total_group_size committee_size) = 5 / 28 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_female_committee_l258_25830


namespace NUMINAMATH_CALUDE_sexual_reproduction_genetic_diversity_l258_25819

/-- Represents a set of genes -/
def GeneticMaterial : Type := Set Nat

/-- Represents an organism with genetic material -/
structure Organism :=
  (genes : GeneticMaterial)

/-- Represents the process of meiosis -/
def meiosis (parent : Organism) : GeneticMaterial :=
  sorry

/-- Represents the process of fertilization -/
def fertilization (gamete1 gamete2 : GeneticMaterial) : Organism :=
  sorry

/-- Theorem stating that sexual reproduction produces offspring with different genetic combinations -/
theorem sexual_reproduction_genetic_diversity 
  (parent1 parent2 : Organism) : 
  ∃ (offspring : Organism), 
    offspring = fertilization (meiosis parent1) (meiosis parent2) ∧
    offspring.genes ≠ parent1.genes ∧
    offspring.genes ≠ parent2.genes :=
  sorry

end NUMINAMATH_CALUDE_sexual_reproduction_genetic_diversity_l258_25819


namespace NUMINAMATH_CALUDE_lawn_mowing_earnings_l258_25821

theorem lawn_mowing_earnings 
  (total_lawns : ℕ) 
  (unmowed_lawns : ℕ) 
  (total_earnings : ℕ) 
  (h1 : total_lawns = 17) 
  (h2 : unmowed_lawns = 9) 
  (h3 : total_earnings = 32) : 
  (total_earnings : ℚ) / ((total_lawns - unmowed_lawns) : ℚ) = 4 := by
sorry

end NUMINAMATH_CALUDE_lawn_mowing_earnings_l258_25821


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l258_25874

/-- Represents the number of books Robert can read in a given time -/
def books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (available_hours : ℕ) : ℕ :=
  (pages_per_hour * available_hours) / pages_per_book

/-- Theorem stating that Robert can read 2 books in 6 hours -/
theorem robert_reading_capacity : books_read 90 270 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l258_25874


namespace NUMINAMATH_CALUDE_photo_arrangements_l258_25838

/-- The number of male students -/
def num_male : Nat := 4

/-- The number of female students -/
def num_female : Nat := 3

/-- The total number of students -/
def total_students : Nat := num_male + num_female

/-- Calculates the number of arrangements with male student A at one end -/
def arrangements_A_at_end : Nat :=
  2 * Nat.factorial (total_students - 1)

/-- Calculates the number of arrangements where female student B is not to the left of female student C -/
def arrangements_B_not_left_of_C : Nat :=
  Nat.factorial total_students / 2

/-- Calculates the number of arrangements where female student B is not at the ends and female student C is not in the middle -/
def arrangements_B_not_ends_C_not_middle : Nat :=
  Nat.factorial (total_students - 1) + 4 * 5 * Nat.factorial (total_students - 2)

theorem photo_arrangements :
  arrangements_A_at_end = 1440 ∧
  arrangements_B_not_left_of_C = 2520 ∧
  arrangements_B_not_ends_C_not_middle = 3120 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_l258_25838


namespace NUMINAMATH_CALUDE_yadav_expenditure_l258_25817

/-- Represents Mr. Yadav's monthly salary in some monetary unit -/
def monthly_salary : ℝ := sorry

/-- Represents the percentage of salary spent on consumable items -/
def consumable_percentage : ℝ := 0.6

/-- Represents the percentage of remaining salary spent on clothes and transport -/
def clothes_transport_percentage : ℝ := 0.5

/-- Represents the yearly savings -/
def yearly_savings : ℝ := 24624

theorem yadav_expenditure :
  let remaining_after_consumables := monthly_salary * (1 - consumable_percentage)
  let clothes_transport_expenditure := remaining_after_consumables * clothes_transport_percentage
  let monthly_savings := yearly_savings / 12
  clothes_transport_expenditure = 2052 := by sorry

end NUMINAMATH_CALUDE_yadav_expenditure_l258_25817


namespace NUMINAMATH_CALUDE_machine_x_production_rate_l258_25841

/-- The number of sprockets produced by both machines -/
def total_sprockets : ℕ := 660

/-- The additional time taken by Machine X compared to Machine B -/
def time_difference : ℕ := 10

/-- The production rate of Machine B relative to Machine X -/
def rate_ratio : ℚ := 11/10

/-- The production rate of Machine X in sprockets per hour -/
def machine_x_rate : ℚ := 6

theorem machine_x_production_rate :
  ∃ (machine_b_rate : ℚ) (time_x time_b : ℚ),
    machine_b_rate = rate_ratio * machine_x_rate ∧
    time_x = time_b + time_difference ∧
    machine_x_rate * time_x = total_sprockets ∧
    machine_b_rate * time_b = total_sprockets :=
by sorry

end NUMINAMATH_CALUDE_machine_x_production_rate_l258_25841


namespace NUMINAMATH_CALUDE_inradius_eq_centroid_height_l258_25862

/-- A non-equilateral triangle with sides a, b, and c, where a + b = 2c -/
structure NonEquilateralTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  non_equilateral : a ≠ b ∨ b ≠ c ∨ a ≠ c
  side_relation : a + b = 2 * c

/-- The inradius of a triangle -/
def inradius (t : NonEquilateralTriangle) : ℝ :=
  sorry

/-- The vertical distance from the base c to the centroid -/
def centroid_height (t : NonEquilateralTriangle) : ℝ :=
  sorry

/-- Theorem stating that the inradius is equal to the vertical distance from the base to the centroid -/
theorem inradius_eq_centroid_height (t : NonEquilateralTriangle) :
  inradius t = centroid_height t :=
sorry

end NUMINAMATH_CALUDE_inradius_eq_centroid_height_l258_25862


namespace NUMINAMATH_CALUDE_parabola_focus_l258_25849

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * x^2 + 4 * x + 1

/-- The focus of a parabola -/
def focus (f : ℝ × ℝ) (p : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c : ℝ), 
    (∀ x y, p x y ↔ y = a * x^2 + b * x + c) ∧
    f = (- b / (2 * a), c - b^2 / (4 * a) - 1 / (4 * a))

/-- Theorem: The focus of the parabola y = -2x^2 + 4x + 1 is (1, 23/8) -/
theorem parabola_focus : focus (1, 23/8) parabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l258_25849


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l258_25844

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if it forms an acute angle of 60° with the y-axis,
    then its eccentricity is 2√3/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_angle : b / a = Real.sqrt 3 / 3) : 
  let e := Real.sqrt (1 + (b / a)^2)
  e = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l258_25844


namespace NUMINAMATH_CALUDE_chandler_wrapping_paper_sales_l258_25800

/-- Chandler's wrapping paper sales problem -/
theorem chandler_wrapping_paper_sales 
  (total_goal : ℕ) 
  (sold_to_grandmother : ℕ) 
  (sold_to_uncle : ℕ) 
  (sold_to_neighbor : ℕ) 
  (h1 : total_goal = 12)
  (h2 : sold_to_grandmother = 3)
  (h3 : sold_to_uncle = 4)
  (h4 : sold_to_neighbor = 3) :
  total_goal - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 2 :=
by sorry

end NUMINAMATH_CALUDE_chandler_wrapping_paper_sales_l258_25800


namespace NUMINAMATH_CALUDE_zachary_crunches_l258_25879

/-- Given that David did 4 crunches and 13 fewer crunches than Zachary,
    prove that Zachary did 17 crunches. -/
theorem zachary_crunches (david_crunches : ℕ) (difference : ℕ) 
  (h1 : david_crunches = 4)
  (h2 : difference = 13) :
  david_crunches + difference = 17 := by
  sorry

end NUMINAMATH_CALUDE_zachary_crunches_l258_25879


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l258_25865

theorem number_puzzle_solution (A B C : ℤ) 
  (sum_eq : A + B = 44)
  (ratio_eq : 5 * A = 6 * B)
  (diff_eq : C = 2 * (A - B)) :
  A = 24 ∧ B = 20 ∧ C = 8 := by
sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l258_25865


namespace NUMINAMATH_CALUDE_max_profit_at_11_l258_25829

/-- The cost price of each item in yuan -/
def cost_price : ℝ := 8

/-- The initial selling price in yuan -/
def initial_price : ℝ := 9

/-- The initial daily sales volume at the initial price -/
def initial_volume : ℝ := 20

/-- The rate at which sales volume decreases per yuan increase in price -/
def volume_decrease_rate : ℝ := 4

/-- The daily sales volume as a function of the selling price -/
def sales_volume (price : ℝ) : ℝ :=
  initial_volume - volume_decrease_rate * (price - initial_price)

/-- The daily profit as a function of the selling price -/
def daily_profit (price : ℝ) : ℝ :=
  sales_volume price * (price - cost_price)

/-- The theorem stating that the daily profit is maximized at 11 yuan -/
theorem max_profit_at_11 :
  ∃ (max_price : ℝ), max_price = 11 ∧
  ∀ (price : ℝ), price ≥ initial_price →
  daily_profit price ≤ daily_profit max_price :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_11_l258_25829


namespace NUMINAMATH_CALUDE_system_solution_l258_25846

theorem system_solution : ∃ (x y z : ℝ), 
  (x + y = -1) ∧ (x + z = 0) ∧ (y + z = 1) ∧ (x = -1) ∧ (y = 0) ∧ (z = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l258_25846


namespace NUMINAMATH_CALUDE_solve_equation_l258_25871

theorem solve_equation : ∃! x : ℝ, 3 * x - 2 * (10 - x) = 5 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l258_25871


namespace NUMINAMATH_CALUDE_buses_passed_count_l258_25847

/-- Represents the frequency of bus departures in minutes -/
def dallas_departure_frequency : ℕ := 60
def houston_departure_frequency : ℕ := 60

/-- Represents the offset of Houston departures from the hour in minutes -/
def houston_departure_offset : ℕ := 45

/-- Represents the trip duration in hours -/
def trip_duration : ℕ := 6

/-- Represents the number of Dallas-bound buses passed by a Houston-bound bus -/
def buses_passed : ℕ := 11

theorem buses_passed_count :
  buses_passed = 11 := by sorry

end NUMINAMATH_CALUDE_buses_passed_count_l258_25847


namespace NUMINAMATH_CALUDE_keith_card_spend_l258_25861

/-- The amount Keith spent on cards -/
def total_spent (digimon_packs : ℕ) (digimon_price : ℚ) (baseball_price : ℚ) : ℚ :=
  digimon_packs * digimon_price + baseball_price

/-- Proof that Keith spent $23.86 on cards -/
theorem keith_card_spend :
  total_spent 4 (4.45 : ℚ) (6.06 : ℚ) = (23.86 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_keith_card_spend_l258_25861


namespace NUMINAMATH_CALUDE_tangent_circle_right_triangle_l258_25843

/-- Given a right triangle DEF with right angle at E, DF = √85, DE = 7, and a circle with center 
    on DE tangent to DF and EF, prove that FQ = 6 where Q is the point where the circle meets DF. -/
theorem tangent_circle_right_triangle (D E F Q : ℝ × ℝ) 
  (h_right_angle : (E.1 - D.1) * (F.1 - D.1) + (E.2 - D.2) * (F.2 - D.2) = 0)
  (h_df : Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2) = Real.sqrt 85)
  (h_de : Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) = 7)
  (h_circle : ∃ (C : ℝ × ℝ), C ∈ Set.Icc D E ∧ 
    dist C D = dist C Q ∧ dist C E = dist C F ∧ dist C Q = dist C F)
  (h_q_on_df : (Q.1 - D.1) * (F.2 - D.2) = (Q.2 - D.2) * (F.1 - D.1)) :
  dist F Q = 6 := by sorry


end NUMINAMATH_CALUDE_tangent_circle_right_triangle_l258_25843


namespace NUMINAMATH_CALUDE_goods_train_speed_calculation_l258_25870

/-- The speed of the man's train in km/h -/
def man_train_speed : ℝ := 60

/-- The length of the goods train in meters -/
def goods_train_length : ℝ := 280

/-- The time it takes for the goods train to pass the man in seconds -/
def passing_time : ℝ := 9

/-- The speed of the goods train in km/h -/
def goods_train_speed : ℝ := 52

theorem goods_train_speed_calculation :
  (man_train_speed + goods_train_speed) * passing_time / 3600 = goods_train_length / 1000 :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_calculation_l258_25870


namespace NUMINAMATH_CALUDE_prime_iff_binomial_divisible_l258_25882

theorem prime_iff_binomial_divisible (n : ℕ) (h : n > 1) : 
  Nat.Prime n ↔ ∀ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 → n ∣ Nat.choose n k := by
  sorry

end NUMINAMATH_CALUDE_prime_iff_binomial_divisible_l258_25882


namespace NUMINAMATH_CALUDE_triangle_cosine_relation_l258_25895

theorem triangle_cosine_relation (A B C : ℝ) (a b c : ℝ) :
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (Real.cos (B-C) * Real.cos A + Real.cos (2*A) = 1 + Real.cos A * Real.cos (B+C)) →
  ((B = C → Real.cos A = 2/3) ∧ (b^2 + c^2) / a^2 = 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_relation_l258_25895


namespace NUMINAMATH_CALUDE_first_grade_enrollment_l258_25856

theorem first_grade_enrollment (a : ℕ) : 
  (200 ≤ a ∧ a ≤ 300) →
  (∃ R : ℕ, a = 25 * R + 10) →
  (∃ L : ℕ, a = 30 * L - 15) →
  a = 285 := by
sorry

end NUMINAMATH_CALUDE_first_grade_enrollment_l258_25856


namespace NUMINAMATH_CALUDE_sin_product_identity_l258_25878

theorem sin_product_identity : 
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) * Real.sin (80 * π / 180) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_identity_l258_25878


namespace NUMINAMATH_CALUDE_equation_solution_l258_25894

theorem equation_solution :
  ∃ x : ℝ, (x - 6) ^ 4 = (1 / 16)⁻¹ ∧ x = 8 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l258_25894


namespace NUMINAMATH_CALUDE_melanie_dimes_count_l258_25857

def final_dimes (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

theorem melanie_dimes_count : final_dimes 7 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_count_l258_25857


namespace NUMINAMATH_CALUDE_trig_identity_l258_25899

theorem trig_identity : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 
  1 / (Real.cos (10 * π / 180) * Real.cos (20 * π / 180)) := by sorry

end NUMINAMATH_CALUDE_trig_identity_l258_25899


namespace NUMINAMATH_CALUDE_train_length_calculation_l258_25876

/-- Proves that given a train and a platform of equal length, if the train crosses the platform
    in one minute at a speed of 144 km/hr, then the length of the train is 1200 meters. -/
theorem train_length_calculation (train_length platform_length : ℝ) 
    (h1 : train_length = platform_length)
    (h2 : train_length + platform_length = 144 * 1000 / 60) : 
    train_length = 1200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l258_25876


namespace NUMINAMATH_CALUDE_f_is_quadratic_l258_25832

/-- A quadratic equation in terms of x is of the form ax² + bx + c = 0, where a ≠ 0 --/
def is_quadratic_in_x (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x² - x + 1 --/
def f (x : ℝ) : ℝ := x^2 - x + 1

/-- Theorem: f(x) = x² - x + 1 is a quadratic equation in terms of x --/
theorem f_is_quadratic : is_quadratic_in_x f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l258_25832


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l258_25809

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (1 - 2 * Complex.I) ∧ 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l258_25809


namespace NUMINAMATH_CALUDE_log_equality_implies_y_value_l258_25877

-- Define the logarithm relationship
def log_relation (m y : ℝ) : Prop :=
  (Real.log y / Real.log m) * (Real.log m / Real.log 7) = 4

-- Theorem statement
theorem log_equality_implies_y_value :
  ∀ m y : ℝ, m > 0 ∧ m ≠ 1 ∧ y > 0 → log_relation m y → y = 2401 :=
by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_y_value_l258_25877


namespace NUMINAMATH_CALUDE_check_amount_proof_l258_25891

theorem check_amount_proof (C : ℝ) 
  (tip_percentage : ℝ) 
  (tip_contribution : ℝ) : 
  tip_percentage = 0.20 → 
  tip_contribution = 40 → 
  tip_percentage * C = tip_contribution → 
  C = 200 := by
sorry

end NUMINAMATH_CALUDE_check_amount_proof_l258_25891


namespace NUMINAMATH_CALUDE_max_concert_tickets_l258_25801

theorem max_concert_tickets (ticket_cost : ℚ) (available_money : ℚ) : 
  ticket_cost = 15 → available_money = 120 → 
  (∃ (n : ℕ), n * ticket_cost ≤ available_money ∧ 
    ∀ (m : ℕ), m * ticket_cost ≤ available_money → m ≤ n) → 
  (∃ (max_tickets : ℕ), max_tickets = 8) :=
by sorry

end NUMINAMATH_CALUDE_max_concert_tickets_l258_25801


namespace NUMINAMATH_CALUDE_adam_orchard_apples_l258_25868

/-- Represents the number of apples Adam collected from his orchard -/
def total_apples (daily_apples : ℕ) (days : ℕ) (remaining_apples : ℕ) : ℕ :=
  daily_apples * days + remaining_apples

/-- Theorem stating the total number of apples Adam collected -/
theorem adam_orchard_apples :
  total_apples 4 30 230 = 350 := by
  sorry

end NUMINAMATH_CALUDE_adam_orchard_apples_l258_25868


namespace NUMINAMATH_CALUDE_cylinder_height_calculation_l258_25851

/-- Given a cylinder with radius 3 units, if increasing the radius by 4 units
    and increasing the height by 10 units both result in the same volume increase,
    then the original height of the cylinder is 2.25 units. -/
theorem cylinder_height_calculation (h : ℝ) : 
  let r := 3
  let new_r := r + 4
  let new_h := h + 10
  let volume := π * r^2 * h
  let volume_after_radius_increase := π * new_r^2 * h
  let volume_after_height_increase := π * r^2 * new_h
  (volume_after_radius_increase - volume = volume_after_height_increase - volume) →
  h = 2.25 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_calculation_l258_25851


namespace NUMINAMATH_CALUDE_simplify_expression_l258_25839

theorem simplify_expression (x : ℝ) : (2*x + 20) + (150*x + 20) = 152*x + 40 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l258_25839


namespace NUMINAMATH_CALUDE_toy_selection_proof_l258_25864

def factorial (n : ℕ) : ℕ := sorry

def combinations (n r : ℕ) : ℕ := 
  factorial n / (factorial r * factorial (n - r))

theorem toy_selection_proof : 
  combinations 10 3 = 120 := by sorry

end NUMINAMATH_CALUDE_toy_selection_proof_l258_25864


namespace NUMINAMATH_CALUDE_four_possible_values_for_D_l258_25883

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem four_possible_values_for_D :
  ∀ (A B C D : ℕ),
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    A < 10 → B < 10 → C < 10 → D < 10 →
    is_even A →
    is_odd B →
    A + B = D →
    C + D = D →
    (∃ (S : Finset ℕ), S.card = 4 ∧ ∀ d, d ∈ S ↔ (∃ a b, is_even a ∧ is_odd b ∧ a + b = d ∧ a < 10 ∧ b < 10 ∧ a ≠ b)) :=
by sorry

end NUMINAMATH_CALUDE_four_possible_values_for_D_l258_25883


namespace NUMINAMATH_CALUDE_height_survey_groups_l258_25825

theorem height_survey_groups (max_height min_height class_interval : ℝ) 
  (h1 : max_height = 173)
  (h2 : min_height = 140)
  (h3 : class_interval = 5) : 
  Int.ceil ((max_height - min_height) / class_interval) = 7 := by
  sorry

end NUMINAMATH_CALUDE_height_survey_groups_l258_25825


namespace NUMINAMATH_CALUDE_dog_food_cans_per_package_l258_25811

/-- Proves that the number of cans in each package of dog food is 5 -/
theorem dog_food_cans_per_package : 
  ∀ (cat_packages dog_packages cat_cans_per_package : ℕ),
    cat_packages = 9 →
    dog_packages = 7 →
    cat_cans_per_package = 10 →
    cat_packages * cat_cans_per_package = dog_packages * 5 + 55 →
    5 = (cat_packages * cat_cans_per_package - 55) / dog_packages := by
  sorry

end NUMINAMATH_CALUDE_dog_food_cans_per_package_l258_25811


namespace NUMINAMATH_CALUDE_estate_value_l258_25806

def estate_problem (total_estate : ℚ) : Prop :=
  let daughters_son_share := (3 : ℚ) / 5 * total_estate
  let first_daughter := (5 : ℚ) / 10 * daughters_son_share
  let second_daughter := (3 : ℚ) / 10 * daughters_son_share
  let son := (2 : ℚ) / 10 * daughters_son_share
  let husband := 2 * son
  let gardener := 600
  let charity := 800
  total_estate = first_daughter + second_daughter + son + husband + gardener + charity

theorem estate_value : 
  ∃ (total_estate : ℚ), estate_problem total_estate ∧ total_estate = 35000 := by
  sorry

end NUMINAMATH_CALUDE_estate_value_l258_25806


namespace NUMINAMATH_CALUDE_banana_distribution_l258_25896

theorem banana_distribution (total : Nat) (friends : Nat) (bananas_per_friend : Nat) :
  total = 36 → friends = 5 → bananas_per_friend = 7 →
  total / friends = bananas_per_friend :=
by sorry

end NUMINAMATH_CALUDE_banana_distribution_l258_25896


namespace NUMINAMATH_CALUDE_travel_time_ratio_l258_25893

theorem travel_time_ratio : 
  let distance : ℝ := 252
  let original_time : ℝ := 6
  let new_speed : ℝ := 28
  let new_time : ℝ := distance / new_speed
  let original_speed : ℝ := distance / original_time
  new_time / original_time = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_travel_time_ratio_l258_25893


namespace NUMINAMATH_CALUDE_parallelogram_area_main_theorem_l258_25888

/-- Represents a parallelogram formed by vertices of smaller triangles inside an equilateral triangle -/
structure Parallelogram where
  m : ℕ  -- Length of one side in terms of unit triangles
  n : ℕ  -- Length of the other side in terms of unit triangles

/-- The main theorem statement -/
theorem parallelogram_area (p : Parallelogram) : 
  (p.m > 1 ∧ p.n > 1) →  -- Sides must be greater than 1 unit triangle
  (p.m + p.n = 6) →      -- Sum of sides is 6 (derived from the 46 triangles condition)
  (p.m * p.n = 8 ∨ p.m * p.n = 9) := by
  sorry

/-- The equilateral triangle ABC -/
def ABC : Set (ℝ × ℝ) := sorry

/-- The set of 400 smaller equilateral triangles -/
def small_triangles : Set (Set (ℝ × ℝ)) := sorry

/-- The condition that the parallelogram is formed by vertices of smaller triangles inside ABC -/
def parallelogram_in_triangle (p : Parallelogram) : Prop := sorry

/-- The condition that the parallelogram sides are parallel to ABC sides -/
def parallel_to_ABC_sides (p : Parallelogram) : Prop := sorry

/-- The condition that exactly 46 smaller triangles have at least one point in common with the parallelogram sides -/
def triangles_touching_sides (p : Parallelogram) : Prop := sorry

/-- The main theorem combining all conditions -/
theorem main_theorem (p : Parallelogram) :
  parallelogram_in_triangle p →
  parallel_to_ABC_sides p →
  triangles_touching_sides p →
  (p.m * p.n = 8 ∨ p.m * p.n = 9) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_main_theorem_l258_25888


namespace NUMINAMATH_CALUDE_rose_puzzle_l258_25869

theorem rose_puzzle : ∃! n : ℕ, 
  300 ≤ n ∧ n ≤ 400 ∧ 
  n % 21 = 13 ∧ 
  n % 15 = 7 ∧ 
  n = 307 := by sorry

end NUMINAMATH_CALUDE_rose_puzzle_l258_25869


namespace NUMINAMATH_CALUDE_mikes_payment_l258_25897

/-- Calculates Mike's out-of-pocket payment for medical procedures -/
theorem mikes_payment (xray_cost : ℝ) (mri_multiplier : ℝ) (insurance_coverage_percent : ℝ) : 
  xray_cost = 250 →
  mri_multiplier = 3 →
  insurance_coverage_percent = 80 →
  let total_cost := xray_cost + mri_multiplier * xray_cost
  let insurance_coverage := (insurance_coverage_percent / 100) * total_cost
  total_cost - insurance_coverage = 200 := by
sorry


end NUMINAMATH_CALUDE_mikes_payment_l258_25897


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_l258_25873

theorem smallest_two_digit_multiple : ∃ n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (∃ k : ℕ, n = 30 * k + 2) ∧
  (∀ m : ℕ, m ≥ 10 ∧ m < 100 ∧ (∃ j : ℕ, m = 30 * j + 2) → m ≥ n) ∧
  n = 32 := by
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_l258_25873


namespace NUMINAMATH_CALUDE_ninth_power_five_and_eleventh_power_five_l258_25837

theorem ninth_power_five_and_eleventh_power_five :
  9^5 = 59149 ∧ 11^5 = 161051 := by
  sorry

#check ninth_power_five_and_eleventh_power_five

end NUMINAMATH_CALUDE_ninth_power_five_and_eleventh_power_five_l258_25837


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l258_25823

/-- Given the conditions of a man's speed in various situations, prove that his speed against the current with wind, waves, and raft is 4 km/hr. -/
theorem mans_speed_against_current (speed_with_current speed_of_current wind_effect wave_effect raft_effect : ℝ)
  (h1 : speed_with_current = 20)
  (h2 : speed_of_current = 5)
  (h3 : wind_effect = 2)
  (h4 : wave_effect = 1)
  (h5 : raft_effect = 3) :
  speed_with_current - speed_of_current - wind_effect - speed_of_current - wave_effect - raft_effect = 4 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l258_25823


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_11_l258_25808

/-- Represents a five-digit number in the form 53A47 -/
def number (A : ℕ) : ℕ := 53000 + A * 100 + 47

/-- Checks if a number is divisible by 11 -/
def divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

theorem five_digit_divisible_by_11 :
  ∃ (A : ℕ), A < 10 ∧ divisible_by_11 (number A) ∧
  ∀ (B : ℕ), B < A → ¬divisible_by_11 (number B) :=
by sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_11_l258_25808


namespace NUMINAMATH_CALUDE_abc_inequality_l258_25816

theorem abc_inequality : 
  let a : ℝ := -(0.3^2)
  let b : ℝ := 3⁻¹
  let c : ℝ := (-1/3)^0
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l258_25816


namespace NUMINAMATH_CALUDE_scientific_notation_of_1206_million_l258_25813

theorem scientific_notation_of_1206_million : 
  ∃ (a : ℝ) (n : ℤ), 1206000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.206 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1206_million_l258_25813


namespace NUMINAMATH_CALUDE_commentator_mistake_l258_25858

def round_robin_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem commentator_mistake (n : ℕ) (x y : ℚ) (h1 : n = 15) :
  ¬(∃ (x y : ℚ),
    (x > 0) ∧
    (y > x) ∧
    (y < 2 * x) ∧
    (3 * x + 13 * y = round_robin_games n)) :=
  sorry

end NUMINAMATH_CALUDE_commentator_mistake_l258_25858


namespace NUMINAMATH_CALUDE_intersection_of_ranges_equality_of_ranges_equality_of_functions_l258_25805

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 4*x + 1

def A₁ : Set ℝ := Set.Icc 1 2
def S₁ : Set ℝ := Set.image f A₁
def T₁ : Set ℝ := Set.image g A₁

def A₂ (m : ℝ) : Set ℝ := Set.Icc 0 m
def S₂ (m : ℝ) : Set ℝ := Set.image f (A₂ m)
def T₂ (m : ℝ) : Set ℝ := Set.image g (A₂ m)

theorem intersection_of_ranges : S₁ ∩ T₁ = {5} := by sorry

theorem equality_of_ranges (m : ℝ) : S₂ m = T₂ m → m = 4 := by sorry

theorem equality_of_functions : 
  {A : Set ℝ | ∀ x ∈ A, f x = g x} ⊆ {{0}, {4}, {0, 4}} := by sorry

end NUMINAMATH_CALUDE_intersection_of_ranges_equality_of_ranges_equality_of_functions_l258_25805


namespace NUMINAMATH_CALUDE_average_cost_before_gratuity_l258_25810

theorem average_cost_before_gratuity 
  (total_people : ℕ) 
  (total_bill : ℚ) 
  (gratuity_rate : ℚ) 
  (h1 : total_people = 9)
  (h2 : total_bill = 756)
  (h3 : gratuity_rate = 1/5) : 
  (total_bill / (1 + gratuity_rate)) / total_people = 70 :=
by sorry

end NUMINAMATH_CALUDE_average_cost_before_gratuity_l258_25810


namespace NUMINAMATH_CALUDE_complex_sum_problem_l258_25835

theorem complex_sum_problem (a b c d e f : ℝ) :
  b = 5 →
  e = -2 * (a + c) →
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) = 4 * Complex.I →
  d + f = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l258_25835


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l258_25889

theorem complex_arithmetic_equality : 
  |(-3) - (-5)| + ((-1/2 : ℚ)^3) / (1/4 : ℚ) * 2 - 6 * ((1/3 : ℚ) - (1/2 : ℚ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l258_25889


namespace NUMINAMATH_CALUDE_wall_space_to_paint_is_560_l258_25859

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents a rectangular feature on a wall (e.g., door or window) -/
structure WallFeature where
  width : ℝ
  height : ℝ

/-- Calculates the total area of wall space to paint in a room -/
def wallSpaceToPaint (room : RoomDimensions) (doorway1 : WallFeature) (window : WallFeature) (doorway2 : WallFeature) : ℝ :=
  let totalWallArea := 2 * (room.width * room.height + room.length * room.height)
  let featureArea := doorway1.width * doorway1.height + window.width * window.height + doorway2.width * doorway2.height
  totalWallArea - featureArea

/-- The main theorem stating that the wall space to paint is 560 square feet -/
theorem wall_space_to_paint_is_560 (room : RoomDimensions) (doorway1 : WallFeature) (window : WallFeature) (doorway2 : WallFeature) :
  room.width = 20 ∧ room.length = 20 ∧ room.height = 8 ∧
  doorway1.width = 3 ∧ doorway1.height = 7 ∧
  window.width = 6 ∧ window.height = 4 ∧
  doorway2.width = 5 ∧ doorway2.height = 7 →
  wallSpaceToPaint room doorway1 window doorway2 = 560 :=
by
  sorry

end NUMINAMATH_CALUDE_wall_space_to_paint_is_560_l258_25859


namespace NUMINAMATH_CALUDE_fraction_power_equality_l258_25866

theorem fraction_power_equality : (81000 ^ 5 : ℕ) / (9000 ^ 5 : ℕ) = 59049 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l258_25866
