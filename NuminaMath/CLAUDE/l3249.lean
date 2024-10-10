import Mathlib

namespace fraction_integer_iff_specific_p_l3249_324954

theorem fraction_integer_iff_specific_p (p : ℕ+) :
  (∃ (n : ℕ+), (3 * p + 25 : ℚ) / (2 * p - 5) = n) ↔ p ∈ ({3, 5, 9, 35} : Set ℕ+) := by
  sorry

end fraction_integer_iff_specific_p_l3249_324954


namespace adidas_to_skechers_ratio_l3249_324998

/-- Proves the ratio of Adidas to Skechers sneakers spending is 1:5 --/
theorem adidas_to_skechers_ratio
  (total_spent : ℕ)
  (nike_to_adidas_ratio : ℕ)
  (adidas_cost : ℕ)
  (clothes_cost : ℕ)
  (h1 : total_spent = 8000)
  (h2 : nike_to_adidas_ratio = 3)
  (h3 : adidas_cost = 600)
  (h4 : clothes_cost = 2600) :
  (adidas_cost : ℚ) / (total_spent - clothes_cost - nike_to_adidas_ratio * adidas_cost - adidas_cost) = 1 / 5 := by
  sorry


end adidas_to_skechers_ratio_l3249_324998


namespace remainder_theorem_l3249_324931

theorem remainder_theorem : ∃ q : ℕ, 3^303 + 303 = q * (3^151 + 3^75 + 1) + 294 := by
  sorry

end remainder_theorem_l3249_324931


namespace garden_minimum_width_l3249_324949

theorem garden_minimum_width :
  ∀ w : ℝ,
  w > 0 →
  w * (w + 12) ≥ 120 →
  w ≥ 6 :=
by sorry

end garden_minimum_width_l3249_324949


namespace repetend_of_three_thirteenths_l3249_324992

/-- The decimal representation of 3/13 has a 6-digit repetend of 230769 -/
theorem repetend_of_three_thirteenths : ∃ (n : ℕ), 
  (3 : ℚ) / 13 = (230769 : ℚ) / 999999 + n / (999999 * 13) := by
  sorry

end repetend_of_three_thirteenths_l3249_324992


namespace complex_product_equality_l3249_324997

theorem complex_product_equality : (3 + 4*Complex.I) * (2 - 3*Complex.I) * (1 + 2*Complex.I) = 20 + 35*Complex.I := by
  sorry

end complex_product_equality_l3249_324997


namespace parabola_directrix_l3249_324985

/-- Given a parabola with equation y² = 2x, its directrix has the equation x = -1/2 -/
theorem parabola_directrix (x y : ℝ) : 
  (y^2 = 2*x) → (∃ (a : ℝ), a = -1/2 ∧ (∀ (x₀ y₀ : ℝ), y₀^2 = 2*x₀ → x₀ = a)) :=
by sorry

end parabola_directrix_l3249_324985


namespace quadratic_expression_equality_l3249_324926

theorem quadratic_expression_equality (x : ℝ) (h : 2 * x^2 + 3 * x + 1 = 10) :
  4 * x^2 + 6 * x + 1 = 19 := by
  sorry

end quadratic_expression_equality_l3249_324926


namespace perfect_square_quadratic_l3249_324970

theorem perfect_square_quadratic (x k : ℝ) : 
  (∃ b : ℝ, ∀ x, x^2 - 20*x + k = (x + b)^2) ↔ k = 100 := by sorry

end perfect_square_quadratic_l3249_324970


namespace sin_2x_value_l3249_324981

theorem sin_2x_value (x : ℝ) (h : Real.sin (x + π/4) = -3/5) : 
  Real.sin (2*x) = -7/25 := by sorry

end sin_2x_value_l3249_324981


namespace jeremy_age_l3249_324976

theorem jeremy_age (total_age : ℕ) (amy_age : ℚ) (chris_age : ℚ) (jeremy_age : ℚ) :
  total_age = 132 →
  amy_age = (1 : ℚ) / 3 * jeremy_age →
  chris_age = 2 * amy_age →
  jeremy_age + amy_age + chris_age = total_age →
  jeremy_age = 66 := by
  sorry

end jeremy_age_l3249_324976


namespace polynomial_roots_l3249_324986

theorem polynomial_roots : ∃ (a b c : ℝ), 
  (a = -1 ∧ b = Real.sqrt 6 ∧ c = -Real.sqrt 6) ∧
  (∀ x : ℝ, x^3 + x^2 - 6*x - 6 = 0 ↔ (x = a ∨ x = b ∨ x = c)) := by
  sorry

end polynomial_roots_l3249_324986


namespace initial_girls_count_l3249_324924

theorem initial_girls_count (total : ℕ) : 
  (total ≠ 0) →
  (total / 2 : ℚ) = (total / 2 : ℕ) →
  ((total / 2 : ℕ) - 5 : ℚ) / total = 2 / 5 →
  total / 2 = 25 := by
  sorry

end initial_girls_count_l3249_324924


namespace train_crossing_time_l3249_324900

/-- The time taken for a train to cross a platform -/
theorem train_crossing_time (train_length : Real) (train_speed_kmph : Real) (platform_length : Real) :
  train_length = 120 ∧ 
  train_speed_kmph = 72 ∧ 
  platform_length = 380.04 →
  (train_length + platform_length) / (train_speed_kmph * 1000 / 3600) = 25.002 := by
  sorry

end train_crossing_time_l3249_324900


namespace laws_in_concept_l3249_324925

/-- The probability that exactly M laws are included in the Concept -/
def prob_exactly_M (K N M : ℕ) (p : ℝ) : ℝ :=
  Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)

/-- The expected number of laws included in the Concept -/
def expected_laws (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

/-- Theorem stating the probability of exactly M laws being included and the expected number of laws -/
theorem laws_in_concept (K N M : ℕ) (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : M ≤ K) :
  (prob_exactly_M K N M p = Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)) ∧
  (expected_laws K N p = K * (1 - (1 - p)^N)) := by
  sorry

#check laws_in_concept

end laws_in_concept_l3249_324925


namespace tangent_line_circle_l3249_324928

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x - y + 3 = 0

/-- The circle equation -/
def circle_equation (x y a : ℝ) : Prop := x^2 + y^2 - 2*x + 2 - a = 0

/-- The theorem statement -/
theorem tangent_line_circle (a : ℝ) :
  (∃ x y : ℝ, line_equation x y ∧ circle_equation x y a ∧
    ∀ x' y' : ℝ, line_equation x' y' → circle_equation x' y' a → (x = x' ∧ y = y')) →
  a = 9 := by sorry

end tangent_line_circle_l3249_324928


namespace smallest_product_of_two_digit_numbers_l3249_324972

-- Define a function to create all possible two-digit numbers from four digits
def twoDigitNumbers (a b c d : Nat) : List (Nat × Nat) :=
  [(10*a + b, 10*c + d), (10*a + c, 10*b + d), (10*a + d, 10*b + c),
   (10*b + a, 10*c + d), (10*b + c, 10*a + d), (10*b + d, 10*a + c),
   (10*c + a, 10*b + d), (10*c + b, 10*a + d), (10*c + d, 10*a + b),
   (10*d + a, 10*b + c), (10*d + b, 10*a + c), (10*d + c, 10*a + b)]

-- Define the theorem
theorem smallest_product_of_two_digit_numbers :
  let digits := [2, 4, 5, 8]
  let products := (twoDigitNumbers 2 4 5 8).map (fun (x, y) => x * y)
  (products.minimum? : Option Nat) = some 1200 := by sorry

end smallest_product_of_two_digit_numbers_l3249_324972


namespace intersection_of_A_and_B_l3249_324956

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 < 0}

-- Define set B
def B : Set ℝ := {x | |x - 2| ≥ 1}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = A_intersect_B := by
  sorry

end intersection_of_A_and_B_l3249_324956


namespace clock_hands_coincidence_time_l3249_324903

/-- Represents the state of a clock's hands -/
structure ClockState where
  minute_angle : ℝ
  hour_angle : ℝ

/-- Represents the movement rates of clock hands -/
structure ClockRates where
  minute_rate : ℝ
  hour_rate : ℝ

/-- Calculates the time taken for clock hands to move from one state to another -/
def time_between_states (initial : ClockState) (final : ClockState) (rates : ClockRates) : ℝ :=
  sorry

theorem clock_hands_coincidence_time :
  let initial_state : ClockState := { minute_angle := 0, hour_angle := 180 }
  let final_state : ClockState := { minute_angle := 0, hour_angle := 0 }
  let rates : ClockRates := { minute_rate := 6, hour_rate := 0.5 }
  let time := time_between_states initial_state final_state rates
  time = 360 ∧ time < 12 * 60 := by sorry

end clock_hands_coincidence_time_l3249_324903


namespace cubic_root_sum_l3249_324984

theorem cubic_root_sum (a b c : ℝ) : 
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) →
  24 * a^3 - 36 * a^2 + 14 * a - 1 = 0 →
  24 * b^3 - 36 * b^2 + 14 * b - 1 = 0 →
  24 * c^3 - 36 * c^2 + 14 * c - 1 = 0 →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 158 / 73 := by
sorry

end cubic_root_sum_l3249_324984


namespace highway_on_map_l3249_324906

/-- Represents the scale of a map as a ratio -/
structure MapScale where
  numerator : ℕ
  denominator : ℕ

/-- Converts kilometers to centimeters -/
def km_to_cm (km : ℕ) : ℕ := km * 100000

/-- Calculates the length on a map given the actual length and map scale -/
def length_on_map (actual_length_km : ℕ) (scale : MapScale) : ℕ :=
  (km_to_cm actual_length_km) * scale.numerator / scale.denominator

/-- Theorem stating that a 155 km highway on a 1:500000 scale map is 31 cm long -/
theorem highway_on_map :
  let actual_length_km : ℕ := 155
  let scale : MapScale := ⟨1, 500000⟩
  length_on_map actual_length_km scale = 31 := by sorry

end highway_on_map_l3249_324906


namespace roots_derivative_sum_negative_l3249_324937

open Real

theorem roots_derivative_sum_negative (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → 
  (a * x₁ - log x₁ = 0) → (a * x₂ - log x₂ = 0) →
  (a - 1 / x₁) + (a - 1 / x₂) < 0 := by
  sorry

end roots_derivative_sum_negative_l3249_324937


namespace boat_speed_ratio_l3249_324921

/-- Proves that the ratio of average speed to still water speed is 42/65 for a boat traveling in a river --/
theorem boat_speed_ratio :
  let still_water_speed : ℝ := 20
  let current_speed : ℝ := 8
  let downstream_distance : ℝ := 10
  let upstream_distance : ℝ := 10
  let downstream_speed : ℝ := still_water_speed + current_speed
  let upstream_speed : ℝ := still_water_speed - current_speed
  let total_time : ℝ := downstream_distance / downstream_speed + upstream_distance / upstream_speed
  let total_distance : ℝ := downstream_distance + upstream_distance
  let average_speed : ℝ := total_distance / total_time
  average_speed / still_water_speed = 42 / 65 := by
  sorry


end boat_speed_ratio_l3249_324921


namespace curve_symmetric_about_y_eq_neg_x_l3249_324944

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop := x * y^2 - x^2 * y = -2

-- Define symmetry about y = -x
def symmetric_about_y_eq_neg_x (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ f (-y) (-x)

-- Theorem statement
theorem curve_symmetric_about_y_eq_neg_x :
  symmetric_about_y_eq_neg_x curve_equation :=
sorry

end curve_symmetric_about_y_eq_neg_x_l3249_324944


namespace inequality_proof_l3249_324974

theorem inequality_proof (a b : ℝ) : 
  (a^2 - 1) * (b^2 - 1) ≥ 0 → a^2 + b^2 - 1 - a^2*b^2 ≤ 0 := by
  sorry

end inequality_proof_l3249_324974


namespace gcd_lcm_multiple_relationship_l3249_324930

theorem gcd_lcm_multiple_relationship (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 6) :
  Nat.gcd a b = b ∧ Nat.lcm a b = a := by
  sorry

end gcd_lcm_multiple_relationship_l3249_324930


namespace expansion_equality_l3249_324987

theorem expansion_equality (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end expansion_equality_l3249_324987


namespace factorial_ratio_l3249_324923

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 11 = 12 := by
  sorry

end factorial_ratio_l3249_324923


namespace odd_painted_faces_6_4_2_l3249_324999

/-- Represents a 3D rectangular block of cubes -/
structure Block :=
  (length : Nat) (width : Nat) (height : Nat)

/-- Counts the number of cubes with an odd number of painted faces in a block -/
def oddPaintedFaces (b : Block) : Nat :=
  sorry

/-- The main theorem: In a 6x4x2 block, 16 cubes have an odd number of painted faces -/
theorem odd_painted_faces_6_4_2 : 
  oddPaintedFaces (Block.mk 6 4 2) = 16 := by
  sorry

end odd_painted_faces_6_4_2_l3249_324999


namespace grandmother_age_l3249_324936

def cody_age : ℕ := 14
def grandmother_age_multiplier : ℕ := 6

theorem grandmother_age : 
  cody_age * grandmother_age_multiplier = 84 := by
  sorry

end grandmother_age_l3249_324936


namespace min_value_greater_than_five_l3249_324912

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 + |x + a - 1| + (a + 1)^2

-- State the theorem
theorem min_value_greater_than_five (a : ℝ) :
  (∀ x, f x a > 5) ↔ a < (-1 - Real.sqrt 14) / 2 ∨ a > Real.sqrt 6 / 2 :=
by sorry

end min_value_greater_than_five_l3249_324912


namespace oak_trees_cut_down_l3249_324971

theorem oak_trees_cut_down (initial_trees : ℕ) (remaining_trees : ℕ) :
  initial_trees = 9 →
  remaining_trees = 7 →
  initial_trees - remaining_trees = 2 :=
by sorry

end oak_trees_cut_down_l3249_324971


namespace or_implies_and_implies_not_equivalent_l3249_324938

theorem or_implies_and_implies_not_equivalent :
  ¬(∀ (A B C : Prop), ((A ∨ B) → C) ↔ ((A ∧ B) → C)) := by
sorry

end or_implies_and_implies_not_equivalent_l3249_324938


namespace magnitude_of_vector_difference_l3249_324973

/-- Given two vectors a and b in a plane with an angle of π/2 between them,
    |a| = 1, and |b| = √3, prove that |2a - b| = √7 -/
theorem magnitude_of_vector_difference (a b : ℝ × ℝ) :
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- angle between a and b is π/2
  (a.1^2 + a.2^2 = 1) →  -- |a| = 1
  (b.1^2 + b.2^2 = 3) →  -- |b| = √3
  ((2 * a.1 - b.1)^2 + (2 * a.2 - b.2)^2 = 7) :=  -- |2a - b| = √7
by sorry

end magnitude_of_vector_difference_l3249_324973


namespace specific_pentagon_perimeter_l3249_324962

/-- Pentagon ABCDE with specific side lengths -/
structure Pentagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  AE : ℝ

/-- The perimeter of a pentagon -/
def perimeter (p : Pentagon) : ℝ :=
  p.AB + p.BC + p.CD + p.DE + p.AE

/-- Theorem: The perimeter of the specific pentagon is 12 -/
theorem specific_pentagon_perimeter :
  ∃ (p : Pentagon),
    p.AB = 2 ∧ p.BC = 2 ∧ p.CD = 2 ∧ p.DE = 2 ∧
    p.AE ^ 2 = (p.AB + p.BC) ^ 2 + (p.CD + p.DE) ^ 2 ∧
    perimeter p = 12 := by
  sorry


end specific_pentagon_perimeter_l3249_324962


namespace sine_function_omega_l3249_324915

/-- Given a function f(x) = 2sin(ωx + π/6) with ω > 0, if it intersects the y-axis at (0, 1) 
    and has two adjacent x-intercepts A and B such that the area of triangle PAB is π, 
    then ω = 1/2 -/
theorem sine_function_omega (ω : ℝ) (f : ℝ → ℝ) (A B : ℝ) : 
  ω > 0 →
  (∀ x, f x = 2 * Real.sin (ω * x + π / 6)) →
  f 0 = 1 →
  f A = 0 →
  f B = 0 →
  A < B →
  (B - A) * 1 / 2 = π →
  ω = 1 / 2 := by
  sorry

end sine_function_omega_l3249_324915


namespace complex_fraction_simplification_l3249_324963

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i^3 * (i + 1)) / (i - 1) = -1 := by
  sorry

end complex_fraction_simplification_l3249_324963


namespace rectangle_dimensions_l3249_324939

/-- Represents the dimensions of a rectangle -/
structure RectDimensions where
  length : ℕ
  width : ℕ

/-- Checks if the given dimensions satisfy the problem conditions -/
def satisfiesConditions (dim : RectDimensions) : Prop :=
  dim.length + dim.width = 11 ∧
  (dim.length = 5 ∧ dim.width = 6) ∨
  (dim.length = 8 ∧ dim.width = 3) ∨
  (dim.length = 4 ∧ dim.width = 7)

theorem rectangle_dimensions :
  ∀ (dim : RectDimensions),
    (2 * (dim.length + dim.width) = 22) →
    (∃ (subRect : RectDimensions),
      subRect.length = 2 ∧ subRect.width = 6 ∧
      subRect.length ≤ dim.length ∧ subRect.width ≤ dim.width) →
    satisfiesConditions dim :=
by sorry

end rectangle_dimensions_l3249_324939


namespace tourist_base_cottages_l3249_324990

theorem tourist_base_cottages :
  ∀ (x : ℕ) (n : ℕ+),
    (2 * x) + x + (n : ℕ) * x ≥ 70 →
    3 * ((n : ℕ) * x) = 2 * x + 25 →
    (2 * x) + x + (n : ℕ) * x = 100 :=
by
  sorry

end tourist_base_cottages_l3249_324990


namespace matrix_solution_l3249_324961

def determinant (a : ℝ) (x : ℝ) : ℝ :=
  (2*x + a) * ((x + a)^2 - x^2) - x * (x*(x + a) - x^2) + x * (x^2 - x*(x + a))

theorem matrix_solution (a : ℝ) (ha : a ≠ 0) :
  {x : ℝ | determinant a x = 0} = {-a/2, a/Real.sqrt 2, -a/Real.sqrt 2} :=
sorry

end matrix_solution_l3249_324961


namespace micah_ate_six_strawberries_l3249_324917

/-- The number of strawberries in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of strawberries Micah picked -/
def dozens_picked : ℕ := 2

/-- The number of strawberries Micah saved for his mom -/
def saved_for_mom : ℕ := 18

/-- The total number of strawberries Micah picked -/
def total_picked : ℕ := dozens_picked * dozen

/-- The number of strawberries Micah ate -/
def eaten_by_micah : ℕ := total_picked - saved_for_mom

theorem micah_ate_six_strawberries : eaten_by_micah = 6 := by
  sorry

end micah_ate_six_strawberries_l3249_324917


namespace ronald_laundry_proof_l3249_324913

/-- The number of days between Tim's laundry sessions -/
def tim_laundry_interval : ℕ := 9

/-- The number of days until Ronald and Tim do laundry on the same day again -/
def next_common_laundry_day : ℕ := 18

/-- The number of days between Ronald's laundry sessions -/
def ronald_laundry_interval : ℕ := 3

theorem ronald_laundry_proof :
  (tim_laundry_interval ∣ next_common_laundry_day) ∧
  (ronald_laundry_interval ∣ next_common_laundry_day) ∧
  (∀ n : ℕ, n ∣ next_common_laundry_day → n ≤ ronald_laundry_interval ∨ ronald_laundry_interval < n) →
  ronald_laundry_interval = 3 :=
by sorry

end ronald_laundry_proof_l3249_324913


namespace merchant_articles_l3249_324995

/-- Represents the number of articles a merchant has -/
def N : ℕ := 20

/-- Represents the cost price of each article -/
def CP : ℝ := 1

/-- Represents the selling price of each article -/
def SP : ℝ := 1.25 * CP

theorem merchant_articles :
  (N * CP = 16 * SP) ∧ (SP = 1.25 * CP) → N = 20 := by
  sorry

end merchant_articles_l3249_324995


namespace import_tax_percentage_l3249_324950

/-- The import tax percentage problem -/
theorem import_tax_percentage 
  (total_value : ℝ) 
  (tax_threshold : ℝ) 
  (tax_paid : ℝ) 
  (h1 : total_value = 2590)
  (h2 : tax_threshold = 1000)
  (h3 : tax_paid = 111.30)
  : (tax_paid / (total_value - tax_threshold)) = 0.07 := by
  sorry

end import_tax_percentage_l3249_324950


namespace projection_matrix_values_l3249_324910

-- Define the matrix P
def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, 20/49],
    ![c, 29/49]]

-- Define the property of being a projection matrix
def is_projection_matrix (M : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  M * M = M

-- Theorem statement
theorem projection_matrix_values :
  ∀ a c : ℚ, is_projection_matrix (P a c) → a = 41/49 ∧ c = 204/1225 :=
by sorry

end projection_matrix_values_l3249_324910


namespace first_chapter_pages_l3249_324919

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  chapter2_pages : ℕ

/-- The number of pages in the first chapter of a book -/
def pages_in_chapter1 (b : Book) : ℕ := b.total_pages - b.chapter2_pages

/-- Theorem stating that for a book with 93 total pages and 33 pages in the second chapter,
    the first chapter has 60 pages -/
theorem first_chapter_pages :
  ∀ (b : Book), b.total_pages = 93 → b.chapter2_pages = 33 → pages_in_chapter1 b = 60 := by
  sorry

end first_chapter_pages_l3249_324919


namespace area_of_region_l3249_324951

/-- The area of the region defined by x^2 + y^2 + 8x - 18y = 0 is 97π -/
theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 97 ∧ 
   A = Real.pi * (Real.sqrt ((x + 4)^2 + (y - 9)^2)) ^ 2 ∧
   x^2 + y^2 + 8*x - 18*y = 0) := by
  sorry

end area_of_region_l3249_324951


namespace subset_sum_exists_l3249_324948

theorem subset_sum_exists (nums : List ℕ) : 
  nums.length = 100 → 
  (∀ n ∈ nums, n ≤ 100) → 
  nums.sum = 200 → 
  ∃ subset : List ℕ, subset ⊆ nums ∧ subset.sum = 100 := by
  sorry

end subset_sum_exists_l3249_324948


namespace binomial_coefficient_21_15_l3249_324901

theorem binomial_coefficient_21_15 
  (h1 : Nat.choose 20 13 = 77520)
  (h2 : Nat.choose 20 14 = 38760)
  (h3 : Nat.choose 22 15 = 203490) :
  Nat.choose 21 15 = 87210 := by
sorry

end binomial_coefficient_21_15_l3249_324901


namespace interval_equinumerosity_l3249_324941

theorem interval_equinumerosity (a : ℝ) (ha : a > 0) :
  ∃ f : Set.Icc 0 1 → Set.Icc 0 a, Function.Bijective f :=
sorry

end interval_equinumerosity_l3249_324941


namespace triangle_inequalities_l3249_324994

/-- Triangle properties and inequalities -/
theorem triangle_inequalities (a b c S h_a h_b h_c r_a r_b r_c r : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0 ∧ h_a > 0 ∧ h_b > 0 ∧ h_c > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ r > 0)
  (h_area : S = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))))
  (h_altitude : h_a = 2 * S / a ∧ h_b = 2 * S / b ∧ h_c = 2 * S / c)
  (h_excircle : (r_a * r_b * r_c)^2 = S^4 / r^2) :
  (S^3 ≤ (Real.sqrt 3 / 4)^3 * (a * b * c)^2) ∧
  ((h_a * h_b * h_c)^(1/3) ≤ 3^(1/4) * Real.sqrt S) ∧
  (3^(1/4) * Real.sqrt S ≤ (r_a * r_b * r_c)^(1/3)) := by
  sorry

end triangle_inequalities_l3249_324994


namespace quadratic_value_at_negative_two_l3249_324940

theorem quadratic_value_at_negative_two (a b : ℝ) :
  (2 * a * 1^2 + b * 1 = 3) → (a * (-2)^2 - b * (-2) = 6) := by
  sorry

end quadratic_value_at_negative_two_l3249_324940


namespace total_cost_separate_tickets_l3249_324927

def adult_ticket_cost : ℕ := 35
def child_ticket_cost : ℕ := 20
def num_adults : ℕ := 2
def num_children : ℕ := 5

theorem total_cost_separate_tickets :
  num_adults * adult_ticket_cost + num_children * child_ticket_cost = 170 := by
  sorry

end total_cost_separate_tickets_l3249_324927


namespace p_necessary_not_sufficient_l3249_324969

-- Define set A as a subset of real numbers
variable (A : Set ℝ)

-- Define proposition p
def p (A : Set ℝ) : Prop :=
  ∃ x ∈ A, x^2 - 2*x - 3 < 0

-- Define proposition q
def q (A : Set ℝ) : Prop :=
  ∀ x ∈ A, x^2 - 2*x - 3 < 0

-- Theorem stating that p is a necessary but not sufficient condition for q
theorem p_necessary_not_sufficient :
  (∀ A : Set ℝ, q A → p A) ∧ (∃ A : Set ℝ, p A ∧ ¬q A) := by sorry

end p_necessary_not_sufficient_l3249_324969


namespace sin_cos_inequality_l3249_324932

theorem sin_cos_inequality (x : ℝ) : -5 ≤ 4 * Real.sin x + 3 * Real.cos x ∧ 4 * Real.sin x + 3 * Real.cos x ≤ 5 := by
  sorry

end sin_cos_inequality_l3249_324932


namespace f_equality_iff_a_half_l3249_324952

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then (4 : ℝ) ^ x else (2 : ℝ) ^ (a - x)

theorem f_equality_iff_a_half (a : ℝ) (h : a ≠ 1) :
  f a (1 - a) = f a (a - 1) ↔ a = 1/2 := by
  sorry

end f_equality_iff_a_half_l3249_324952


namespace no_fourteen_consecutive_integers_exist_twentyone_consecutive_integers_l3249_324918

/-- Defines a function that checks if a number is divisible by any prime in a given range -/
def divisible_by_prime_in_range (n : ℕ) (lower upper : ℕ) : Prop :=
  ∃ p, Prime p ∧ lower ≤ p ∧ p ≤ upper ∧ p ∣ n

/-- Theorem stating that there do not exist 14 consecutive positive integers
    each divisible by at least one prime p where 2 ≤ p ≤ 11 -/
theorem no_fourteen_consecutive_integers : ¬ ∃ start : ℕ, ∀ k : ℕ, k < 14 →
  divisible_by_prime_in_range (start + k) 2 11 := by sorry

/-- Theorem stating that there exist 21 consecutive positive integers
    each divisible by at least one prime p where 2 ≤ p ≤ 13 -/
theorem exist_twentyone_consecutive_integers : ∃ start : ℕ, ∀ k : ℕ, k < 21 →
  divisible_by_prime_in_range (start + k) 2 13 := by sorry

end no_fourteen_consecutive_integers_exist_twentyone_consecutive_integers_l3249_324918


namespace function_value_given_cube_l3249_324978

theorem function_value_given_cube (x : ℝ) (h : x^3 = 8) :
  (x - 1) * (x + 1) * (x^2 + x + 1) = 21 := by
  sorry

end function_value_given_cube_l3249_324978


namespace defective_shipped_percentage_l3249_324955

/-- The percentage of units with Type A defects in the first stage -/
def type_a_defect_rate : ℝ := 0.07

/-- The percentage of units with Type B defects in the second stage -/
def type_b_defect_rate : ℝ := 0.08

/-- The percentage of Type A defective units that are shipped for sale -/
def type_a_ship_rate : ℝ := 0.03

/-- The percentage of Type B defective units that are shipped for sale -/
def type_b_ship_rate : ℝ := 0.06

/-- The total percentage of defective units (Type A or B) that are shipped for sale -/
def total_defective_shipped_rate : ℝ :=
  type_a_defect_rate * type_a_ship_rate + type_b_defect_rate * type_b_ship_rate

theorem defective_shipped_percentage :
  total_defective_shipped_rate = 0.0069 := by sorry

end defective_shipped_percentage_l3249_324955


namespace initial_tax_rate_proof_l3249_324980

def annual_income : ℝ := 48000
def new_tax_rate : ℝ := 30
def tax_savings : ℝ := 7200

theorem initial_tax_rate_proof :
  ∃ (initial_rate : ℝ),
    initial_rate > 0 ∧
    initial_rate < 100 ∧
    (initial_rate / 100 * annual_income) - (new_tax_rate / 100 * annual_income) = tax_savings ∧
    initial_rate = 45 := by
  sorry

end initial_tax_rate_proof_l3249_324980


namespace some_number_value_l3249_324942

theorem some_number_value (x : ℝ) : 40 + 5 * 12 / (x / 3) = 41 → x = 180 := by
  sorry

end some_number_value_l3249_324942


namespace games_missed_l3249_324947

/-- Given that Benny's high school played 39 baseball games and he attended 14 games,
    prove that the number of games Benny missed is 25. -/
theorem games_missed (total_games : ℕ) (games_attended : ℕ) (h1 : total_games = 39) (h2 : games_attended = 14) :
  total_games - games_attended = 25 := by
  sorry

end games_missed_l3249_324947


namespace adjacent_angles_theorem_l3249_324958

/-- Given two adjacent angles forming a straight line, where one angle is 4x and the other is x, 
    prove that x = 18°. -/
theorem adjacent_angles_theorem (x : ℝ) : 
  (4 * x + x = 180) → x = 18 := by sorry

end adjacent_angles_theorem_l3249_324958


namespace fairCoinDifference_l3249_324964

def fairCoinProbability : ℚ := 1 / 2

def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

def probabilityThreeHeads : ℚ :=
  binomialProbability 4 3 fairCoinProbability

def probabilityFourHeads : ℚ :=
  fairCoinProbability^4

theorem fairCoinDifference :
  probabilityThreeHeads - probabilityFourHeads = 3 / 16 := by
  sorry

end fairCoinDifference_l3249_324964


namespace math_exam_questions_l3249_324904

theorem math_exam_questions (english_questions : ℕ) (english_time : ℕ) (math_time : ℕ) (extra_time_per_question : ℕ) : 
  english_questions = 30 →
  english_time = 60 →
  math_time = 90 →
  extra_time_per_question = 4 →
  (math_time / (english_time / english_questions + extra_time_per_question) : ℕ) = 15 := by
sorry

end math_exam_questions_l3249_324904


namespace least_number_divisible_by_four_primes_l3249_324934

theorem least_number_divisible_by_four_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
   p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
   p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
     q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
     q₁ ∣ m ∧ q₂ ∣ m ∧ q₃ ∣ m ∧ q₄ ∣ m) → 
    n ≤ m) ∧
  n = 210 :=
by sorry

end least_number_divisible_by_four_primes_l3249_324934


namespace cosine_sum_upper_bound_l3249_324943

theorem cosine_sum_upper_bound (α β γ : Real) 
  (h : Real.sin α + Real.sin β + Real.sin γ ≥ 2) : 
  Real.cos α + Real.cos β + Real.cos γ ≤ Real.sqrt 5 := by
  sorry

end cosine_sum_upper_bound_l3249_324943


namespace complex_exp_13pi_over_3_l3249_324920

theorem complex_exp_13pi_over_3 : Complex.exp (13 * π * Complex.I / 3) = (1 / 2 : ℂ) + Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end complex_exp_13pi_over_3_l3249_324920


namespace higher_speed_is_two_l3249_324907

-- Define the runners
structure Runner :=
  (blocks : ℕ)
  (minutes : ℕ)

-- Define the speed calculation function
def speed (r : Runner) : ℚ :=
  r.blocks / r.minutes

-- Define Tiffany and Moses
def tiffany : Runner := ⟨6, 3⟩
def moses : Runner := ⟨12, 8⟩

-- Theorem: The higher average speed is 2 blocks per minute
theorem higher_speed_is_two :
  max (speed tiffany) (speed moses) = 2 := by
  sorry

end higher_speed_is_two_l3249_324907


namespace complex_fraction_product_l3249_324959

theorem complex_fraction_product (a b : ℝ) : 
  (1 + 7 * Complex.I) / (2 - Complex.I) = Complex.mk a b → a * b = -3 := by
  sorry

end complex_fraction_product_l3249_324959


namespace no_solution_exists_l3249_324953

theorem no_solution_exists : ¬∃ x : ℝ, (|x^2 - 14*x + 40| = 3) ∧ (x^2 - 14*x + 45 = 0) := by
  sorry

end no_solution_exists_l3249_324953


namespace computer_sticker_price_l3249_324935

theorem computer_sticker_price : 
  ∀ (sticker_price : ℝ),
  (sticker_price * 0.85 - 90 = sticker_price * 0.75 - 15) →
  sticker_price = 750 := by
sorry

end computer_sticker_price_l3249_324935


namespace rectangle_area_l3249_324911

/-- The area of a rectangle with perimeter 40 and length twice its width -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) (h2 : 6 * w = 40) : w * (2 * w) = 800 / 9 := by
  sorry

end rectangle_area_l3249_324911


namespace shortest_chord_length_l3249_324905

/-- The shortest chord length of the intersection between a line and a circle -/
theorem shortest_chord_length (m : ℝ) : 
  let l := {(x, y) : ℝ × ℝ | 2 * m * x - y - 8 * m - 3 = 0}
  let C := {(x, y) : ℝ × ℝ | x^2 + y^2 - 6 * x + 12 * y + 20 = 0}
  ∃ (chord_length : ℝ), 
    chord_length = 2 * Real.sqrt 15 ∧ 
    ∀ (other_length : ℝ), 
      (∃ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ l ∧ p ∈ C ∧ q ∈ C ∧ 
        other_length = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) →
      other_length ≥ chord_length :=
by
  sorry

end shortest_chord_length_l3249_324905


namespace compare_negative_fractions_l3249_324929

theorem compare_negative_fractions : -2/3 > -5/7 := by
  sorry

end compare_negative_fractions_l3249_324929


namespace u_value_l3249_324975

/-- A line passing through points (2, 8), (4, 14), (6, 20), and (18, u) -/
structure Line where
  -- Define the slope of the line
  slope : ℝ
  -- Define the y-intercept of the line
  intercept : ℝ
  -- Ensure the line passes through (2, 8)
  point1 : 8 = slope * 2 + intercept
  -- Ensure the line passes through (4, 14)
  point2 : 14 = slope * 4 + intercept
  -- Ensure the line passes through (6, 20)
  point3 : 20 = slope * 6 + intercept

/-- The u-coordinate of the point (18, u) on the line -/
def u (l : Line) : ℝ := l.slope * 18 + l.intercept

/-- Theorem stating that u = 56 for the given line -/
theorem u_value (l : Line) : u l = 56 := by
  sorry

end u_value_l3249_324975


namespace prism_volume_l3249_324967

/-- The volume of a right rectangular prism with given face areas and one side length -/
theorem prism_volume (side_area front_area bottom_area : ℝ) (known_side : ℝ)
  (h_side : side_area = 20)
  (h_front : front_area = 12)
  (h_bottom : bottom_area = 15)
  (h_known : known_side = 5) :
  ∃ (a b c : ℝ),
    a * b = side_area ∧
    b * c = front_area ∧
    a * c = bottom_area ∧
    b = known_side ∧
    a * b * c = 75 := by
  sorry

end prism_volume_l3249_324967


namespace root_sum_reciprocal_diff_l3249_324945

-- Define the polynomial
def p (x : ℝ) : ℝ := 45 * x^3 - 75 * x^2 + 33 * x - 2

-- Define the theorem
theorem root_sum_reciprocal_diff (a b c : ℝ) :
  p a = 0 → p b = 0 → p c = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  0 < a → a < 1 →
  0 < b → b < 1 →
  0 < c → c < 1 →
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = 4 / 3 := by
sorry

end root_sum_reciprocal_diff_l3249_324945


namespace kingsleys_friends_l3249_324993

theorem kingsleys_friends (chairs_per_trip : ℕ) (total_trips : ℕ) (total_chairs : ℕ) :
  chairs_per_trip = 5 →
  total_trips = 10 →
  total_chairs = 250 →
  (total_chairs / (chairs_per_trip * total_trips)) - 1 = 4 := by
  sorry

end kingsleys_friends_l3249_324993


namespace expression_value_l3249_324957

theorem expression_value (x y : ℝ) (h : x / (2 * y) = 3 / 2) :
  (7 * x + 6 * y) / (x - 2 * y) = 27 := by
  sorry

end expression_value_l3249_324957


namespace incorrect_calculation_l3249_324991

theorem incorrect_calculation (x : ℝ) : (-3 * x)^2 ≠ 6 * x^2 := by
  sorry

end incorrect_calculation_l3249_324991


namespace final_orchid_count_l3249_324933

/-- The number of orchids in a vase after adding more -/
def orchids_in_vase (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

theorem final_orchid_count : orchids_in_vase 3 4 = 7 := by
  sorry

end final_orchid_count_l3249_324933


namespace smallest_exponent_of_ten_l3249_324914

theorem smallest_exponent_of_ten (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 2012 → 
  a.factorial * b.factorial * c.factorial = m * 10^n → 
  ¬(10 ∣ m) → 
  (∀ k : ℕ, k < n → ∃ (m' : ℕ), a.factorial * b.factorial * c.factorial = m' * 10^k ∧ 10 ∣ m') →
  n = 501 := by
sorry

end smallest_exponent_of_ten_l3249_324914


namespace seven_balls_three_boxes_l3249_324966

/-- The number of ways to place distinguishable balls into distinguishable boxes -/
def place_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 2187 ways to place 7 distinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : place_balls 7 3 = 2187 := by
  sorry

end seven_balls_three_boxes_l3249_324966


namespace johnny_works_four_hours_on_third_job_l3249_324908

/-- Represents Johnny's work schedule and earnings --/
structure WorkSchedule where
  hours_job1 : ℕ
  rate_job1 : ℕ
  hours_job2 : ℕ
  rate_job2 : ℕ
  rate_job3 : ℕ
  days : ℕ
  total_earnings : ℕ

/-- Calculates the number of hours worked on the third job each day --/
def hours_job3_per_day (w : WorkSchedule) : ℕ :=
  let daily_earnings_job12 := w.hours_job1 * w.rate_job1 + w.hours_job2 * w.rate_job2
  let total_earnings_job12 := daily_earnings_job12 * w.days
  let total_earnings_job3 := w.total_earnings - total_earnings_job12
  total_earnings_job3 / (w.rate_job3 * w.days)

/-- Theorem stating that given Johnny's work schedule, he works 4 hours on the third job each day --/
theorem johnny_works_four_hours_on_third_job (w : WorkSchedule)
  (h1 : w.hours_job1 = 3)
  (h2 : w.rate_job1 = 7)
  (h3 : w.hours_job2 = 2)
  (h4 : w.rate_job2 = 10)
  (h5 : w.rate_job3 = 12)
  (h6 : w.days = 5)
  (h7 : w.total_earnings = 445) :
  hours_job3_per_day w = 4 := by
  sorry

end johnny_works_four_hours_on_third_job_l3249_324908


namespace unique_positive_number_sum_with_square_l3249_324965

theorem unique_positive_number_sum_with_square : ∃! x : ℝ, x > 0 ∧ x^2 + x = 156 := by
  sorry

end unique_positive_number_sum_with_square_l3249_324965


namespace digit_B_is_three_l3249_324983

/-- Represents a digit from 1 to 7 -/
def Digit := Fin 7

/-- Represents the set of points A, B, C, D, E, F -/
structure Points where
  A : Digit
  B : Digit
  C : Digit
  D : Digit
  E : Digit
  F : Digit
  distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
             B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
             C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
             D ≠ E ∧ D ≠ F ∧
             E ≠ F

/-- The sum of digits along each line -/
def lineSums (p : Points) : ℕ :=
  (p.A.val + p.B.val + p.C.val + 1) +
  (p.A.val + p.E.val + p.F.val + 1) +
  (p.C.val + p.D.val + p.E.val + 1) +
  (p.B.val + p.D.val + 1) +
  (p.B.val + p.F.val + 1)

theorem digit_B_is_three (p : Points) (h : lineSums p = 51) : p.B.val + 1 = 3 := by
  sorry

end digit_B_is_three_l3249_324983


namespace cory_fruit_arrangements_l3249_324916

def fruit_arrangements (total : ℕ) (apples oranges bananas : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas)

theorem cory_fruit_arrangements :
  fruit_arrangements 9 4 2 2 = 3780 :=
by sorry

end cory_fruit_arrangements_l3249_324916


namespace expression_evaluation_l3249_324988

theorem expression_evaluation : 
  Real.sqrt ((16^10 + 8^10 + 2^30) / (16^4 + 8^11 + 2^20)) = 1/2 := by
  sorry

end expression_evaluation_l3249_324988


namespace power_sum_equality_l3249_324996

theorem power_sum_equality : (-1)^43 + 2^(2^3 + 5^2 - 7^2) = -(65535 / 65536) := by
  sorry

end power_sum_equality_l3249_324996


namespace parabola_equation_l3249_324909

-- Define the parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the line x - y + 2 = 0
def focus_line (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the conditions for the parabola
def parabola_conditions (p : Parabola) : Prop :=
  -- Vertex at origin
  p.equation 0 0
  -- Axis of symmetry is a coordinate axis
  ∧ (∀ x y : ℝ, p.equation x y → p.equation x (-y) ∨ p.equation (-x) y)
  -- Focus on the line x - y + 2 = 0
  ∧ ∃ fx fy : ℝ, focus_line fx fy ∧ 
    ((∀ x y : ℝ, p.equation x y ↔ (x - fx)^2 + (y - fy)^2 = (x + fx)^2 + (y + fy)^2)
    ∨ (∀ x y : ℝ, p.equation x y ↔ (x - fx)^2 + (y - fy)^2 = (x - fx)^2 + (y + fy)^2))

-- Theorem statement
theorem parabola_equation (p : Parabola) (h : parabola_conditions p) :
  (∀ x y : ℝ, p.equation x y ↔ y^2 = -8*x) ∨ (∀ x y : ℝ, p.equation x y ↔ x^2 = 8*y) :=
sorry

end parabola_equation_l3249_324909


namespace binomial_difference_divisibility_l3249_324968

theorem binomial_difference_divisibility (p n : ℕ) (hp : Prime p) (hn : n > p) :
  ∃ k : ℤ, (Nat.choose (n + p - 1) p : ℤ) - (Nat.choose n p : ℤ) = k * n :=
sorry

end binomial_difference_divisibility_l3249_324968


namespace right_triangle_hypotenuse_l3249_324989

/-- Given a right triangle, if rotating it about one leg produces a cone of volume 972π cm³
    and rotating it about the other leg produces a cone of volume 1458π cm³,
    then the length of the hypotenuse is 12√5 cm. -/
theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  (1/3) * π * a * b^2 = 972 * π →
  (1/3) * π * b * a^2 = 1458 * π →
  c = 12 * Real.sqrt 5 := by
  sorry

end right_triangle_hypotenuse_l3249_324989


namespace line_parallel_to_parallel_plane_l3249_324979

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between planes
variable (planeParallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the containment relation between a line and a plane
variable (lineInPlane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane
  (a : Line) (α β : Plane)
  (h1 : planeParallel α β)
  (h2 : lineInPlane a α) :
  lineParallelPlane a β :=
sorry

end line_parallel_to_parallel_plane_l3249_324979


namespace number_added_before_division_l3249_324946

theorem number_added_before_division (x : ℤ) : 
  (∃ k : ℤ, x = 82 * k + 5) → 
  (∃ n : ℤ, ∃ m : ℤ, x + n = 41 * m + 18) → 
  (∃ n : ℤ, x + n ≡ 18 [ZMOD 41] ∧ n = 5) :=
by sorry

end number_added_before_division_l3249_324946


namespace concentric_circles_ratio_l3249_324982

theorem concentric_circles_ratio 
  (r R : ℝ) 
  (a b c : ℝ) 
  (h_positive : 0 < r ∧ 0 < R ∧ 0 < a ∧ 0 < b ∧ 0 < c)
  (h_r_less_R : r < R)
  (h_area_ratio : (π * R^2 - π * r^2) / (π * R^2) = a / (b + c)) :
  R / r = Real.sqrt a / Real.sqrt (b + c - a) := by
  sorry

end concentric_circles_ratio_l3249_324982


namespace intersection_M_N_l3249_324902

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 2}
def N : Set ℝ := {x : ℝ | x^2 - 25 < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Icc 2 5 := by
  sorry

end intersection_M_N_l3249_324902


namespace tile_problem_l3249_324960

theorem tile_problem (total_tiles : ℕ) : 
  (∃ n : ℕ, total_tiles = n^2 + 36 ∧ total_tiles = (n + 1)^2 + 3) → 
  total_tiles = 292 := by
sorry

end tile_problem_l3249_324960


namespace ducks_cows_relationship_l3249_324922

/-- Represents a group of ducks and cows -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (group : AnimalGroup) : ℕ := 2 * group.ducks + 4 * group.cows

/-- The total number of heads in the group -/
def totalHeads (group : AnimalGroup) : ℕ := group.ducks + group.cows

/-- The theorem stating the relationship between ducks and cows -/
theorem ducks_cows_relationship (group : AnimalGroup) :
  totalLegs group = 3 * totalHeads group + 26 → group.cows = group.ducks + 26 := by
  sorry


end ducks_cows_relationship_l3249_324922


namespace parade_vehicles_l3249_324977

theorem parade_vehicles (b t q : ℕ) : 
  b + t + q = 12 →
  2*b + 3*t + 4*q = 35 →
  q = 5 :=
by sorry

end parade_vehicles_l3249_324977
