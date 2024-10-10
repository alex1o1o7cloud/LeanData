import Mathlib

namespace line_perp_plane_and_line_parallel_plane_implies_lines_perpendicular_l999_99978

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_perp_plane_and_line_parallel_plane_implies_lines_perpendicular
  (m n : Line) (α : Plane) :
  perpendicular m α → parallel n α → perpendicularLines m n :=
sorry

end line_perp_plane_and_line_parallel_plane_implies_lines_perpendicular_l999_99978


namespace line_parallel_perpendicular_l999_99991

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular n α) : 
  perpendicular m α :=
sorry

end line_parallel_perpendicular_l999_99991


namespace age_problem_l999_99953

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 27 →
  b = 10 := by
sorry

end age_problem_l999_99953


namespace complex_fraction_simplification_l999_99999

/-- Proves that (8-15i)/(3+4i) = -36/25 - 77/25*i -/
theorem complex_fraction_simplification :
  (8 - 15 * Complex.I) / (3 + 4 * Complex.I) = -36/25 - 77/25 * Complex.I := by
  sorry

end complex_fraction_simplification_l999_99999


namespace odd_function_implies_a_zero_l999_99995

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 1) * (x + a)

-- State the theorem
theorem odd_function_implies_a_zero :
  (∀ x, f a x = -f a (-x)) → a = 0 :=
by
  sorry

end odd_function_implies_a_zero_l999_99995


namespace quadratic_sign_l999_99909

/-- A quadratic function of the form f(x) = x^2 + x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + x + c

theorem quadratic_sign (c : ℝ) (p : ℝ) 
  (h1 : f c 0 > 0) 
  (h2 : f c p < 0) : 
  f c (p + 1) > 0 := by
  sorry

end quadratic_sign_l999_99909


namespace sqrt_sum_abs_equal_six_l999_99971

theorem sqrt_sum_abs_equal_six :
  Real.sqrt 2 + Real.sqrt 16 + |Real.sqrt 2 - 2| = 6 := by
  sorry

end sqrt_sum_abs_equal_six_l999_99971


namespace smallest_surface_area_of_glued_cubes_smallest_surface_area_proof_l999_99926

/-- The smallest possible surface area of a polyhedron formed by gluing three cubes with volumes 1, 8, and 27 at their faces. -/
theorem smallest_surface_area_of_glued_cubes : ℝ :=
  let cube1 : ℝ := 1
  let cube2 : ℝ := 8
  let cube3 : ℝ := 27
  let surface_area : ℝ := 72
  surface_area

/-- Proof that the smallest possible surface area of a polyhedron formed by gluing three cubes with volumes 1, 8, and 27 at their faces is 72. -/
theorem smallest_surface_area_proof :
  smallest_surface_area_of_glued_cubes = 72 := by
  sorry

end smallest_surface_area_of_glued_cubes_smallest_surface_area_proof_l999_99926


namespace walter_age_theorem_l999_99960

def walter_age_1999 (walter_age_1994 grandmother_age_1994 birth_year_sum : ℕ) : Prop :=
  walter_age_1994 * 2 = grandmother_age_1994 ∧
  (1994 - walter_age_1994) + (1994 - grandmother_age_1994) = birth_year_sum ∧
  walter_age_1994 + (1999 - 1994) = 55

theorem walter_age_theorem : 
  ∃ (walter_age_1994 grandmother_age_1994 : ℕ), 
    walter_age_1999 walter_age_1994 grandmother_age_1994 3838 :=
by
  sorry

end walter_age_theorem_l999_99960


namespace ride_to_total_ratio_l999_99938

def total_money : ℚ := 30
def dessert_cost : ℚ := 5
def money_left : ℚ := 10

theorem ride_to_total_ratio : 
  (total_money - dessert_cost - money_left) / total_money = 1 / 2 := by
  sorry

end ride_to_total_ratio_l999_99938


namespace shoulder_width_conversion_l999_99914

/-- Converts centimeters to millimeters -/
def cm_to_mm (cm : ℝ) : ℝ := cm * 10

theorem shoulder_width_conversion :
  let cm_per_m : ℝ := 100
  let mm_per_m : ℝ := 1000
  let shoulder_width_cm : ℝ := 45
  cm_to_mm shoulder_width_cm = 450 := by
  sorry

end shoulder_width_conversion_l999_99914


namespace hyperbola_equation_proof_l999_99948

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  /-- The general form of the hyperbola is x²/a² - y²/b² = 1 -/
  a : ℝ
  b : ℝ
  /-- One focus of the hyperbola is at (2,0) -/
  focus_x : a = 2
  /-- The equations of the asymptotes are y = ±√3x -/
  asymptote_slope : b / a = Real.sqrt 3

/-- The equation of the hyperbola with given properties -/
def hyperbola_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, x^2 - y^2 / 3 = 1 ↔ x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Theorem stating that the hyperbola with given properties has the equation x² - y²/3 = 1 -/
theorem hyperbola_equation_proof (h : Hyperbola) : hyperbola_equation h := by
  sorry

end hyperbola_equation_proof_l999_99948


namespace rhombus_area_in_square_l999_99968

/-- The area of a rhombus formed by the intersection of two equilateral triangles in a square -/
theorem rhombus_area_in_square (s : ℝ) (h : s = 4) : 
  let triangle_height := s * Real.sqrt 3 / 2
  let rhombus_diagonal1 := 2 * triangle_height - s
  let rhombus_diagonal2 := s
  rhombus_diagonal1 * rhombus_diagonal2 / 2 = 8 * Real.sqrt 3 - 8 := by
  sorry

#check rhombus_area_in_square

end rhombus_area_in_square_l999_99968


namespace no_rational_solution_l999_99986

theorem no_rational_solution :
  ¬∃ (a b c d : ℚ) (n : ℕ), (a + b * Real.sqrt 2)^(2*n) + (c + d * Real.sqrt 2)^(2*n) = 5 + 4 * Real.sqrt 2 := by
  sorry

end no_rational_solution_l999_99986


namespace fundraiser_item_price_l999_99963

theorem fundraiser_item_price 
  (num_brownie_students : ℕ) 
  (num_cookie_students : ℕ) 
  (num_donut_students : ℕ) 
  (brownies_per_student : ℕ) 
  (cookies_per_student : ℕ) 
  (donuts_per_student : ℕ) 
  (total_amount_raised : ℚ) : 
  num_brownie_students = 30 →
  num_cookie_students = 20 →
  num_donut_students = 15 →
  brownies_per_student = 12 →
  cookies_per_student = 24 →
  donuts_per_student = 12 →
  total_amount_raised = 2040 →
  (total_amount_raised / (num_brownie_students * brownies_per_student + 
                          num_cookie_students * cookies_per_student + 
                          num_donut_students * donuts_per_student) : ℚ) = 2 := by
  sorry

end fundraiser_item_price_l999_99963


namespace chessboard_coloring_theorem_l999_99998

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents a coloring of the chessboard -/
def Coloring := Square → Bool

/-- Checks if three squares form a trimino -/
def isTrimino (s1 s2 s3 : Square) : Prop := sorry

/-- Counts the number of red squares in a coloring -/
def countRedSquares (c : Coloring) : Nat := sorry

/-- Checks if a coloring has no red trimino -/
def hasNoRedTrimino (c : Coloring) : Prop := sorry

/-- Checks if every trimino in a coloring has at least one red square -/
def everyTriminoHasRed (c : Coloring) : Prop := sorry

theorem chessboard_coloring_theorem :
  (∃ c : Coloring, hasNoRedTrimino c ∧ countRedSquares c = 32) ∧
  (∀ c : Coloring, hasNoRedTrimino c → countRedSquares c ≤ 32) ∧
  (∃ c : Coloring, everyTriminoHasRed c ∧ countRedSquares c = 32) ∧
  (∀ c : Coloring, everyTriminoHasRed c → countRedSquares c ≥ 32) := by
  sorry

end chessboard_coloring_theorem_l999_99998


namespace total_interest_earned_l999_99937

def initial_investment : ℝ := 1500
def annual_interest_rate : ℝ := 0.12
def time_period : ℕ := 4

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem total_interest_earned :
  let final_amount := compound_interest initial_investment annual_interest_rate time_period
  final_amount - initial_investment = 862.2 := by
  sorry

end total_interest_earned_l999_99937


namespace constant_term_of_equation_l999_99928

/-- The constant term of a quadratic equation ax^2 + bx + c = 0 is c -/
def constant_term (a b c : ℝ) : ℝ := c

theorem constant_term_of_equation :
  constant_term 3 1 5 = 5 := by sorry

end constant_term_of_equation_l999_99928


namespace overlapping_sectors_area_l999_99915

/-- The area of the overlapping region of two 45° sectors in a circle with radius 15 -/
theorem overlapping_sectors_area (r : ℝ) (angle : ℝ) : 
  r = 15 → angle = 45 → 
  2 * (angle / 360 * π * r^2 - 1/2 * r^2 * Real.sin (angle * π / 180)) = 225/4 * (π - 2 * Real.sqrt 2) := by
  sorry

end overlapping_sectors_area_l999_99915


namespace fraction_repeating_block_length_l999_99977

/-- The number of digits in the smallest repeating block of the decimal expansion of 5/7 -/
def repeating_block_length : ℕ := 6

/-- The fraction we're considering -/
def fraction : ℚ := 5 / 7

theorem fraction_repeating_block_length :
  repeating_block_length = 6 ∧ 
  ∀ n : ℕ, n < repeating_block_length → 
    ∃ k : ℕ, fraction * 10^repeating_block_length - fraction * 10^n = k :=
sorry

end fraction_repeating_block_length_l999_99977


namespace smaller_mold_radius_l999_99924

/-- The radius of a smaller hemisphere-shaped mold when a large hemisphere-shaped bowl
    with radius 1 foot is evenly distributed into 64 congruent smaller molds. -/
theorem smaller_mold_radius : ℝ → ℝ → ℝ → Prop :=
  fun (large_radius : ℝ) (num_molds : ℝ) (small_radius : ℝ) =>
    large_radius = 1 ∧
    num_molds = 64 ∧
    (2/3 * Real.pi * large_radius^3) = (num_molds * (2/3 * Real.pi * small_radius^3)) →
    small_radius = 1/4

/-- Proof of the smaller_mold_radius theorem. -/
lemma prove_smaller_mold_radius : smaller_mold_radius 1 64 (1/4) := by
  sorry

end smaller_mold_radius_l999_99924


namespace hundreds_digit_of_13_pow_2023_l999_99929

theorem hundreds_digit_of_13_pow_2023 : 13^2023 % 1000 = 99 := by sorry

end hundreds_digit_of_13_pow_2023_l999_99929


namespace special_set_property_l999_99900

/-- A set of points in ℝ³ that intersects every plane but has finite intersection with each plane -/
def SpecialSet : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | ∃ t : ℝ, x = t^5 ∧ y = t^3 ∧ z = t}

/-- Definition of a plane in ℝ³ -/
def Plane (a b c d : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | a * x + b * y + c * z + d = 0}

theorem special_set_property :
  ∃ S : Set (ℝ × ℝ × ℝ),
    (∀ a b c d : ℝ, (Plane a b c d ∩ S).Nonempty) ∧
    (∀ a b c d : ℝ, (Plane a b c d ∩ S).Finite) :=
by
  use SpecialSet
  sorry

end special_set_property_l999_99900


namespace square_area_decrease_l999_99903

theorem square_area_decrease (s : ℝ) (h : s > 0) :
  let initial_area := s^2
  let new_side := s * 0.9
  let new_area := new_side * s
  (initial_area - new_area) / initial_area * 100 = 19 := by
  sorry

end square_area_decrease_l999_99903


namespace total_miles_walked_l999_99982

def monday_miles : ℕ := 9
def tuesday_miles : ℕ := 9

theorem total_miles_walked : monday_miles + tuesday_miles = 18 := by
  sorry

end total_miles_walked_l999_99982


namespace natural_numbers_with_special_last_digit_l999_99989

def last_digit (n : ℕ) : ℕ := n % 10

def satisfies_condition (n : ℕ) : Prop :=
  n ≠ 0 ∧ n = 2016 * (last_digit n)

theorem natural_numbers_with_special_last_digit :
  {n : ℕ | satisfies_condition n} = {4032, 8064, 12096, 16128} :=
by sorry

end natural_numbers_with_special_last_digit_l999_99989


namespace M_properties_l999_99965

def M (n : ℕ+) : ℤ := (-2) ^ n.val

theorem M_properties :
  (M 5 + M 6 = 32) ∧
  (2 * M 2015 + M 2016 = 0) ∧
  (∀ n : ℕ+, 2 * M n + M (n + 1) = 0) := by
  sorry

end M_properties_l999_99965


namespace ellipse_hyperbola_semi_axes_product_l999_99905

/-- Given an ellipse and a hyperbola with specific foci, prove the product of their semi-axes -/
theorem ellipse_hyperbola_semi_axes_product (c d : ℝ) : 
  (∀ (x y : ℝ), x^2/c^2 + y^2/d^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ (x y : ℝ), x^2/c^2 - y^2/d^2 = 1 → (x = 8 ∧ y = 0) ∨ (x = -8 ∧ y = 0)) →
  |c * d| = Real.sqrt 868.5 := by
sorry

end ellipse_hyperbola_semi_axes_product_l999_99905


namespace quadratic_inequality_solution_set_l999_99936

theorem quadratic_inequality_solution_set (c : ℝ) (h : c > 1) :
  {x : ℝ | x^2 - (c + 1/c)*x + 1 > 0} = {x : ℝ | x < 1/c ∨ x > c} := by
sorry

end quadratic_inequality_solution_set_l999_99936


namespace birds_joined_l999_99967

theorem birds_joined (initial_birds : ℕ) (final_birds : ℕ) (initial_storks : ℕ) :
  let birds_joined := final_birds - initial_birds
  birds_joined = 6 :=
by
  sorry

end birds_joined_l999_99967


namespace philip_farm_animals_l999_99913

/-- The number of animals on Philip's farm -/
def total_animals (cows ducks pigs : ℕ) : ℕ := cows + ducks + pigs

/-- The number of cows on Philip's farm -/
def number_of_cows : ℕ := 20

/-- The number of ducks on Philip's farm -/
def number_of_ducks : ℕ := number_of_cows + (number_of_cows / 2)

/-- The number of pigs on Philip's farm -/
def number_of_pigs : ℕ := (number_of_cows + number_of_ducks) / 5

theorem philip_farm_animals :
  total_animals number_of_cows number_of_ducks number_of_pigs = 60 := by
  sorry

end philip_farm_animals_l999_99913


namespace f_derivative_l999_99952

noncomputable def f (x : ℝ) : ℝ :=
  (5^x * (2 * Real.sin (2*x) + Real.cos (2*x) * Real.log 5)) / (4 + (Real.log 5)^2)

theorem f_derivative (x : ℝ) :
  deriv f x = 5^x * Real.cos (2*x) :=
by sorry

end f_derivative_l999_99952


namespace equality_of_areas_l999_99919

theorem equality_of_areas (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  (∃ r : Real, r > 0 ∧ 
    (r^2 * θ / 2 = r^2 * Real.tan θ / 2 - r^2 * θ / 2)) ↔ 
  Real.tan θ = 2 * θ := by
sorry

end equality_of_areas_l999_99919


namespace andre_total_cost_l999_99906

/-- Calculates the total cost of Andre's purchases including sales tax -/
def total_cost (treadmill_price : ℝ) (treadmill_discount : ℝ) 
                (plate_price : ℝ) (plate_discount : ℝ) (num_plates : ℕ)
                (sales_tax : ℝ) : ℝ :=
  let discounted_treadmill := treadmill_price * (1 - treadmill_discount)
  let discounted_plates := plate_price * num_plates * (1 - plate_discount)
  let subtotal := discounted_treadmill + discounted_plates
  subtotal * (1 + sales_tax)

/-- Theorem stating that Andre's total cost is $1120.29 -/
theorem andre_total_cost :
  total_cost 1350 0.30 60 0.15 2 0.07 = 1120.29 := by
  sorry

end andre_total_cost_l999_99906


namespace susanas_chocolate_chips_l999_99966

theorem susanas_chocolate_chips 
  (viviana_chocolate : ℕ) 
  (susana_chocolate : ℕ) 
  (viviana_vanilla : ℕ) 
  (susana_vanilla : ℕ) 
  (h1 : viviana_chocolate = susana_chocolate + 5)
  (h2 : susana_vanilla = (3 * viviana_vanilla) / 4)
  (h3 : viviana_vanilla = 20)
  (h4 : viviana_chocolate + susana_chocolate + viviana_vanilla + susana_vanilla = 90) :
  susana_chocolate = 25 := by
sorry

end susanas_chocolate_chips_l999_99966


namespace five_letter_word_count_l999_99958

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The number of five-letter words that begin and end with the same letter
    and have a vowel as the third letter -/
def word_count : ℕ := alphabet_size * alphabet_size * vowel_count * alphabet_size

theorem five_letter_word_count : word_count = 87880 := by
  sorry

end five_letter_word_count_l999_99958


namespace inscribed_circle_in_tangent_quadrilateral_l999_99961

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle with four vertices -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Represents a quadrilateral formed by tangent lines -/
structure TangentQuadrilateral where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ

/-- Function to check if two circles intersect -/
def circlesIntersect (c1 c2 : Circle) : Prop := sorry

/-- Function to check if a quadrilateral is tangential (can have an inscribed circle) -/
def isTangentialQuadrilateral (quad : TangentQuadrilateral) : Prop := sorry

/-- Main theorem statement -/
theorem inscribed_circle_in_tangent_quadrilateral 
  (rect : Rectangle) 
  (circleA circleB circleC circleD : Circle)
  (quad : TangentQuadrilateral) :
  circleA.center = rect.A ∧
  circleB.center = rect.B ∧
  circleC.center = rect.C ∧
  circleD.center = rect.D ∧
  ¬(circlesIntersect circleA circleB) ∧
  ¬(circlesIntersect circleA circleC) ∧
  ¬(circlesIntersect circleA circleD) ∧
  ¬(circlesIntersect circleB circleC) ∧
  ¬(circlesIntersect circleB circleD) ∧
  ¬(circlesIntersect circleC circleD) ∧
  circleA.radius + circleC.radius = circleB.radius + circleD.radius →
  isTangentialQuadrilateral quad :=
by sorry

end inscribed_circle_in_tangent_quadrilateral_l999_99961


namespace completing_square_equivalence_l999_99975

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) := by
  sorry

end completing_square_equivalence_l999_99975


namespace touching_balls_theorem_l999_99997

/-- Represents a spherical ball with a given radius -/
structure Ball where
  radius : ℝ

/-- Represents two touching balls on the ground -/
structure TouchingBalls where
  ball1 : Ball
  ball2 : Ball
  contactHeight : ℝ

/-- The radius of the other ball given the conditions -/
def otherBallRadius (balls : TouchingBalls) : ℝ := 6

theorem touching_balls_theorem (balls : TouchingBalls) 
  (h1 : balls.ball1.radius = 4)
  (h2 : balls.contactHeight = 6) :
  otherBallRadius balls = balls.ball2.radius :=
sorry

end touching_balls_theorem_l999_99997


namespace gray_eyed_brunettes_l999_99988

theorem gray_eyed_brunettes (total : ℕ) (green_eyed_blondes : ℕ) (brunettes : ℕ) (gray_eyed : ℕ)
  (h1 : total = 60)
  (h2 : green_eyed_blondes = 20)
  (h3 : brunettes = 35)
  (h4 : gray_eyed = 25) :
  total - brunettes - green_eyed_blondes = gray_eyed - (total - brunettes) + green_eyed_blondes :=
by
  sorry

#check gray_eyed_brunettes

end gray_eyed_brunettes_l999_99988


namespace inequality_proof_l999_99930

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  Real.sqrt (1 + 2*a) + Real.sqrt (1 + 2*b) ≤ 2 * Real.sqrt 2 := by
  sorry

end inequality_proof_l999_99930


namespace stream_speed_is_three_l999_99944

/-- Represents the scenario of a rower traveling upstream and downstream -/
structure RiverJourney where
  distance : ℝ
  normalSpeedDiff : ℝ
  tripleSpeedDiff : ℝ

/-- Calculates the stream speed given a RiverJourney -/
def calculateStreamSpeed (journey : RiverJourney) : ℝ :=
  sorry

/-- Theorem stating that the stream speed is 3 for the given conditions -/
theorem stream_speed_is_three (journey : RiverJourney)
  (h1 : journey.distance = 21)
  (h2 : journey.normalSpeedDiff = 4)
  (h3 : journey.tripleSpeedDiff = 0.5) :
  calculateStreamSpeed journey = 3 := by
  sorry

end stream_speed_is_three_l999_99944


namespace elderly_arrangements_proof_l999_99987

def number_of_arrangements (n_volunteers : ℕ) (n_elderly : ℕ) : ℕ :=
  (n_volunteers.factorial) * 
  (n_volunteers - 1) * 
  (n_elderly.factorial)

theorem elderly_arrangements_proof :
  number_of_arrangements 4 2 = 144 :=
by sorry

end elderly_arrangements_proof_l999_99987


namespace new_arithmetic_mean_l999_99917

/-- Given a set of 60 numbers with arithmetic mean 42, prove that removing 50 and 60
    and increasing each remaining number by 2 results in a new arithmetic mean of 43.55 -/
theorem new_arithmetic_mean (S : Finset ℝ) (sum_S : ℝ) : 
  S.card = 60 →
  sum_S = S.sum id →
  sum_S / 60 = 42 →
  50 ∈ S →
  60 ∈ S →
  let S' := S.erase 50 ⊔ S.erase 60
  let sum_S' := S'.sum (fun x => x + 2)
  sum_S' / 58 = 43.55 := by
sorry

end new_arithmetic_mean_l999_99917


namespace log_equation_solution_l999_99983

theorem log_equation_solution :
  ∃! x : ℝ, x > 0 ∧ Real.log x + Real.log (x + 1) = 2 ∧ x = (-1 + Real.sqrt 401) / 2 := by
  sorry

end log_equation_solution_l999_99983


namespace first_year_after_2100_digit_sum_15_l999_99957

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Check if a year is after 2100 -/
def is_after_2100 (year : ℕ) : Prop :=
  year > 2100

/-- First year after 2100 with digit sum 15 -/
def first_year_after_2100_with_digit_sum_15 : ℕ := 2139

theorem first_year_after_2100_digit_sum_15 :
  (is_after_2100 first_year_after_2100_with_digit_sum_15) ∧
  (sum_of_digits first_year_after_2100_with_digit_sum_15 = 15) ∧
  (∀ y : ℕ, is_after_2100 y ∧ y < first_year_after_2100_with_digit_sum_15 →
    sum_of_digits y ≠ 15) :=
sorry

end first_year_after_2100_digit_sum_15_l999_99957


namespace last_integer_before_100_l999_99962

def sequence_term (n : ℕ) : ℕ := (16777216 : ℕ) / 2^n

theorem last_integer_before_100 :
  ∃ n : ℕ, sequence_term n = 64 ∧ sequence_term (n + 1) < 100 :=
sorry

end last_integer_before_100_l999_99962


namespace right_triangle_perpendicular_bisector_l999_99979

theorem right_triangle_perpendicular_bisector 
  (A B C D : ℝ × ℝ) 
  (right_angle_at_A : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 75)
  (AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 100)
  (D_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2))
  (AD_perp_BC : (D.1 - A.1) * (C.1 - B.1) + (D.2 - A.2) * (C.2 - B.2) = 0) :
  Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 45 := by
sorry

end right_triangle_perpendicular_bisector_l999_99979


namespace fraction_meaningful_l999_99901

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x - 1) / (x - 2)) ↔ x ≠ 2 := by
sorry

end fraction_meaningful_l999_99901


namespace unique_six_digit_square_l999_99970

/-- Checks if a number has all digits different --/
def has_different_digits (n : Nat) : Bool :=
  sorry

/-- Checks if digits in a number are in ascending order --/
def digits_ascending (n : Nat) : Bool :=
  sorry

/-- The unique six-digit perfect square with ascending, different digits --/
theorem unique_six_digit_square : 
  ∃! n : Nat, 
    100000 ≤ n ∧ n < 1000000 ∧  -- six-digit number
    has_different_digits n ∧ 
    digits_ascending n ∧ 
    ∃ m : Nat, n = m^2 :=
by
  sorry

end unique_six_digit_square_l999_99970


namespace tom_apples_l999_99956

/-- The number of apples Phillip has -/
def phillip_apples : ℕ := 40

/-- The number of apples Ben has more than Phillip -/
def ben_extra_apples : ℕ := 8

/-- The fraction of Ben's apples that Tom has -/
def tom_fraction : ℚ := 3 / 8

/-- Theorem stating that Tom has 18 apples -/
theorem tom_apples : ℕ := by sorry

end tom_apples_l999_99956


namespace multiple_exists_l999_99955

theorem multiple_exists (n : ℕ) (S : Finset ℕ) : 
  S ⊆ Finset.range (2 * n + 1) →
  S.card = n + 1 →
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
sorry

end multiple_exists_l999_99955


namespace train_length_train_length_correct_l999_99945

/-- Represents the scenario of two people walking alongside a moving train --/
structure TrainScenario where
  train_speed : ℝ
  walking_speed : ℝ
  person_a_distance : ℝ
  person_b_distance : ℝ
  (train_speed_positive : train_speed > 0)
  (walking_speed_positive : walking_speed > 0)
  (person_a_distance_positive : person_a_distance > 0)
  (person_b_distance_positive : person_b_distance > 0)
  (person_a_distance_eq : person_a_distance = 45)
  (person_b_distance_eq : person_b_distance = 30)

/-- The theorem stating that given the conditions, the train length is 180 meters --/
theorem train_length (scenario : TrainScenario) : ℝ :=
  180

/-- The main theorem proving that the train length is correct --/
theorem train_length_correct (scenario : TrainScenario) :
  train_length scenario = 180 := by
  sorry

end train_length_train_length_correct_l999_99945


namespace symmetric_points_sum_power_l999_99974

/-- Two points symmetric about the y-axis in a Cartesian coordinate system -/
structure SymmetricPoints where
  m : ℝ
  n : ℝ
  symmetric : m + 4 = 0 ∧ n = 3

/-- The theorem stating that for symmetric points A(m,3) and B(4,n), (m+n)^2023 = -1 -/
theorem symmetric_points_sum_power (p : SymmetricPoints) : (p.m + p.n)^2023 = -1 := by
  sorry

end symmetric_points_sum_power_l999_99974


namespace total_combinations_eq_nine_l999_99954

/-- The number of characters available to choose from. -/
def num_characters : ℕ := 3

/-- The number of cars available to choose from. -/
def num_cars : ℕ := 3

/-- The total number of possible combinations when choosing one character and one car. -/
def total_combinations : ℕ := num_characters * num_cars

/-- Theorem stating that the total number of combinations is 9. -/
theorem total_combinations_eq_nine : total_combinations = 9 := by
  sorry

end total_combinations_eq_nine_l999_99954


namespace samantha_bedtime_l999_99927

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  deriving Repr

/-- Calculates the bedtime given wake-up time and sleep duration -/
def calculateBedtime (wakeUpTime : Time) (sleepDuration : Nat) : Time :=
  let totalMinutes := wakeUpTime.hour * 60 + wakeUpTime.minute
  let bedtimeMinutes := (totalMinutes - sleepDuration * 60 + 24 * 60) % (24 * 60)
  { hour := bedtimeMinutes / 60, minute := bedtimeMinutes % 60 }

theorem samantha_bedtime :
  let wakeUpTime : Time := { hour := 11, minute := 0 }
  let sleepDuration : Nat := 6
  calculateBedtime wakeUpTime sleepDuration = { hour := 5, minute := 0 } := by
  sorry

end samantha_bedtime_l999_99927


namespace pencil_boxes_count_l999_99964

theorem pencil_boxes_count (pencils_per_box : ℝ) (total_pencils : ℕ) 
  (h1 : pencils_per_box = 648.0) 
  (h2 : total_pencils = 2592) : 
  ↑total_pencils / pencils_per_box = 4 := by
  sorry

end pencil_boxes_count_l999_99964


namespace constant_grid_function_l999_99993

theorem constant_grid_function 
  (f : ℤ × ℤ → ℕ) 
  (h : ∀ x y : ℤ, 4 * f (x, y) = f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) : 
  ∀ p q : ℤ × ℤ, f p = f q :=
sorry

end constant_grid_function_l999_99993


namespace unique_hour_conversion_l999_99940

theorem unique_hour_conversion : 
  ∃! n : ℕ, 
    (∃ x y : ℕ, x < 10 ∧ y < 10 ∧ n = 234000 + x * 1000 + y * 100) ∧ 
    (n % 3600 = 0) ∧
    (∃ h : ℕ, n = h * 3600) :=
by
  sorry

end unique_hour_conversion_l999_99940


namespace pie_slices_yesterday_l999_99973

theorem pie_slices_yesterday (total : ℕ) (today : ℕ) (yesterday : ℕ) : 
  total = 7 → today = 2 → yesterday = total - today → yesterday = 5 := by
  sorry

end pie_slices_yesterday_l999_99973


namespace scaling_transformation_result_l999_99916

/-- The scaling transformation applied to a point (x, y) -/
def scaling (x y : ℝ) : ℝ × ℝ := (x, 3 * y)

/-- The original curve C: x^2 + 9y^2 = 9 -/
def original_curve (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9

/-- The transformed curve -/
def transformed_curve (x' y' : ℝ) : Prop := x'^2 + y'^2 = 9

/-- Theorem stating that the scaling transformation of the original curve
    results in the transformed curve -/
theorem scaling_transformation_result :
  ∀ x y : ℝ, original_curve x y →
  let (x', y') := scaling x y
  transformed_curve x' y' := by
  sorry

end scaling_transformation_result_l999_99916


namespace division_of_decimals_l999_99912

theorem division_of_decimals : (0.45 : ℚ) / (0.005 : ℚ) = 90 := by sorry

end division_of_decimals_l999_99912


namespace milk_water_solution_volume_l999_99925

theorem milk_water_solution_volume 
  (initial_milk_percentage : ℝ) 
  (final_milk_percentage : ℝ) 
  (added_water : ℝ) 
  (initial_milk_percentage_value : initial_milk_percentage = 0.84)
  (final_milk_percentage_value : final_milk_percentage = 0.58)
  (added_water_value : added_water = 26.9) : 
  ∃ (initial_volume : ℝ), 
    initial_volume > 0 ∧ 
    initial_milk_percentage * initial_volume / (initial_volume + added_water) = final_milk_percentage ∧
    initial_volume = 60 := by
  sorry

end milk_water_solution_volume_l999_99925


namespace not_passed_implies_scored_less_than_90_percent_l999_99923

-- Define the proposition for scoring at least 90% on the final exam
def scored_at_least_90_percent (student : Type) : Prop := sorry

-- Define the proposition for passing the course
def passed_course (student : Type) : Prop := sorry

-- State the given condition
axiom condition (student : Type) : passed_course student → scored_at_least_90_percent student

-- State the theorem to be proved
theorem not_passed_implies_scored_less_than_90_percent (student : Type) :
  ¬(passed_course student) → ¬(scored_at_least_90_percent student) := by sorry

end not_passed_implies_scored_less_than_90_percent_l999_99923


namespace find_number_l999_99981

theorem find_number : ∃ N : ℚ, (5/6 : ℚ) * N = (5/16 : ℚ) * N + 50 → N = 96 := by
  sorry

end find_number_l999_99981


namespace smallest_common_factor_l999_99922

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m < 43 → gcd (5 * m - 3) (11 * m + 4) = 1) ∧ 
  gcd (5 * 43 - 3) (11 * 43 + 4) > 1 := by
  sorry

end smallest_common_factor_l999_99922


namespace sum_of_roots_squared_diff_sum_of_roots_eq_fourteen_l999_99950

theorem sum_of_roots_squared_diff (a b : ℝ) :
  (∃ x y : ℝ, (x - a)^2 = b^2 ∧ (y - a)^2 = b^2 ∧ x + y = 2 * a) :=
by
  sorry

theorem sum_of_roots_eq_fourteen :
  let roots := {x : ℝ | (x - 7)^2 = 16}
  (∃ x y : ℝ, x ∈ roots ∧ y ∈ roots ∧ x + y = 14) :=
by
  sorry

end sum_of_roots_squared_diff_sum_of_roots_eq_fourteen_l999_99950


namespace visit_all_points_prob_one_l999_99910

/-- Represents a one-dimensional random walk --/
structure RandomWalk where
  p : ℝ  -- Probability of moving right or left
  r : ℝ  -- Probability of staying in place
  prob_sum : p + p + r = 1  -- Sum of probabilities equals 1

/-- The probability of eventually reaching any point from any starting position --/
def eventual_visit_prob (rw : RandomWalk) : ℝ → ℝ := sorry

/-- Theorem stating that if p > 0, the probability of visiting any point is 1 --/
theorem visit_all_points_prob_one (rw : RandomWalk) (h : rw.p > 0) :
  ∀ x, eventual_visit_prob rw x = 1 := by sorry

end visit_all_points_prob_one_l999_99910


namespace johns_number_is_eleven_l999_99908

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_switch (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem johns_number_is_eleven :
  ∃! x : ℕ, is_two_digit x ∧
    82 ≤ digit_switch (5 * x + 13) ∧
    digit_switch (5 * x + 13) ≤ 86 ∧
    x = 11 := by sorry

end johns_number_is_eleven_l999_99908


namespace all_are_siblings_l999_99969

-- Define a finite type with 7 elements to represent the boys
inductive Boy : Type
  | B1 | B2 | B3 | B4 | B5 | B6 | B7

-- Define the sibling relation
def is_sibling : Boy → Boy → Prop := sorry

-- State the theorem
theorem all_are_siblings :
  (∀ b : Boy, ∃ (s : Finset Boy), s.card ≥ 3 ∧ ∀ s' ∈ s, is_sibling b s') →
  (∀ b1 b2 : Boy, is_sibling b1 b2) :=
sorry

end all_are_siblings_l999_99969


namespace joes_gym_people_l999_99933

/-- The number of people in Joe's Gym during Bethany's shift --/
theorem joes_gym_people (W A : ℕ) : 
  W + A + 5 + 2 - 3 - 4 + 2 = 20 → W + A = 18 := by
  sorry

#check joes_gym_people

end joes_gym_people_l999_99933


namespace problem_solution_l999_99931

/-- S(n, k) denotes the number of coefficients in the expansion of (x+1)^n that are not divisible by k -/
def S (n k : ℕ) : ℕ := sorry

theorem problem_solution :
  (S 2012 3 = 324) ∧ (2012 ∣ S (2012^2011) 2011) := by sorry

end problem_solution_l999_99931


namespace certain_number_proof_l999_99985

theorem certain_number_proof (X : ℝ) : 
  X / 3 = (169.4915254237288 / 100) * 236 → X = 1200 := by
  sorry

end certain_number_proof_l999_99985


namespace separating_chord_length_l999_99946

/-- Represents a hexagon inscribed in a circle -/
structure InscribedHexagon where
  -- The lengths of the sides
  side_lengths : Fin 6 → ℝ
  -- Condition that alternating sides have lengths 5 and 4
  alternating_sides : ∀ i : Fin 6, side_lengths i = if i % 2 = 0 then 5 else 4

/-- The chord that separates the hexagon into two trapezoids -/
def separating_chord (h : InscribedHexagon) : ℝ := sorry

/-- Theorem stating the length of the separating chord -/
theorem separating_chord_length (h : InscribedHexagon) :
  separating_chord h = 180 / 49 := by sorry

end separating_chord_length_l999_99946


namespace maruti_car_price_increase_l999_99980

theorem maruti_car_price_increase (P S : ℝ) (x : ℝ) 
  (h1 : P > 0) (h2 : S > 0) : 
  (P * (1 + x / 100) * (S * 0.8) = P * S * 1.04) → x = 30 := by
  sorry

end maruti_car_price_increase_l999_99980


namespace triangle_abc_properties_l999_99932

/-- Triangle ABC with given conditions -/
structure TriangleABC where
  -- Vertex A coordinates
  A : ℝ × ℝ
  -- Equation of line containing median CM on side AB
  median_CM_eq : ℝ → ℝ → ℝ
  -- Equation of line containing altitude BH on side AC
  altitude_BH_eq : ℝ → ℝ → ℝ
  -- Conditions
  h_A : A = (5, 1)
  h_median_CM : ∀ x y, median_CM_eq x y = 2*x - y - 5
  h_altitude_BH : ∀ x y, altitude_BH_eq x y = x - 2*y - 5

/-- Main theorem about Triangle ABC -/
theorem triangle_abc_properties (t : TriangleABC) :
  -- 1. Coordinates of vertex C
  ∃ C : ℝ × ℝ, C = (4, 3) ∧
  -- 2. Length of AC
  Real.sqrt ((C.1 - t.A.1)^2 + (C.2 - t.A.2)^2) = Real.sqrt 5 ∧
  -- 3. Equation of line BC
  ∃ BC_eq : ℝ → ℝ → ℝ, (∀ x y, BC_eq x y = 6*x - 5*y - 9) :=
by sorry

end triangle_abc_properties_l999_99932


namespace combined_mean_of_two_sets_l999_99951

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 7 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 20 →
  (set1_count * set1_mean + set2_count * set2_mean) / (set1_count + set2_count) = 265 / 15 := by
  sorry

end combined_mean_of_two_sets_l999_99951


namespace polynomial_division_theorem_l999_99959

theorem polynomial_division_theorem (x : ℝ) :
  ∃ (q r : ℝ), 5*x^4 - 3*x^3 + 7*x^2 - 9*x + 12 = (x - 3) * (5*x^3 + 12*x^2 + 43*x + 120) + 372 :=
by
  sorry

end polynomial_division_theorem_l999_99959


namespace refrigerator_transport_cost_l999_99939

/-- Calculate the transport cost for a refrigerator purchase --/
theorem refrigerator_transport_cost 
  (purchase_price : ℕ) 
  (discount_rate : ℚ) 
  (installation_cost : ℕ) 
  (profit_rate : ℚ) 
  (selling_price : ℕ) 
  (h1 : purchase_price = 14500)
  (h2 : discount_rate = 1/5)
  (h3 : installation_cost = 250)
  (h4 : profit_rate = 1/10)
  (h5 : selling_price = 20350) : 
  ∃ (transport_cost : ℕ), transport_cost = 3375 :=
by
  sorry

#check refrigerator_transport_cost

end refrigerator_transport_cost_l999_99939


namespace power_sum_difference_l999_99934

theorem power_sum_difference : 3^(1+2+3+4) - (3^1 + 3^2 + 3^3 + 3^4) - 3^5 = 58686 := by
  sorry

end power_sum_difference_l999_99934


namespace geometric_mean_of_4_and_9_l999_99949

-- Define the geometric mean
def geometric_mean (a c : ℝ) : Set ℝ :=
  {b : ℝ | a * c = b^2}

-- Theorem statement
theorem geometric_mean_of_4_and_9 :
  geometric_mean 4 9 = {6, -6} := by
  sorry

end geometric_mean_of_4_and_9_l999_99949


namespace prism_height_l999_99911

/-- Represents a triangular prism with a regular triangular base -/
structure TriangularPrism where
  baseSideLength : ℝ
  totalEdgeLength : ℝ
  height : ℝ

/-- Theorem: The height of a specific triangular prism -/
theorem prism_height (p : TriangularPrism) 
  (h1 : p.baseSideLength = 10)
  (h2 : p.totalEdgeLength = 84) :
  p.height = 8 := by
  sorry

end prism_height_l999_99911


namespace exists_max_in_finite_list_l999_99918

theorem exists_max_in_finite_list : 
  ∀ (L : List ℝ), L.length = 1000 → ∃ (m : ℝ), ∀ (x : ℝ), x ∈ L → x ≤ m :=
by sorry

end exists_max_in_finite_list_l999_99918


namespace tangent_line_sum_l999_99943

def tangent_line (f : ℝ → ℝ) (a : ℝ) (m : ℝ) (b : ℝ) :=
  ∀ x, f a + m * (x - a) = m * x + b

theorem tangent_line_sum (f : ℝ → ℝ) :
  tangent_line f 5 (-1) 8 → f 5 + (deriv f) 5 = 2 := by
  sorry

end tangent_line_sum_l999_99943


namespace total_lives_after_third_level_l999_99902

/-- Game rules for calculating lives --/
def game_lives : ℕ → ℕ :=
  let initial_lives := 2
  let first_level_gain := 6 / 2
  let second_level_gain := 11 - 3
  let third_level_multiplier := 2
  fun level =>
    if level = 0 then
      initial_lives
    else if level = 1 then
      initial_lives + first_level_gain
    else if level = 2 then
      initial_lives + first_level_gain + second_level_gain
    else
      initial_lives + first_level_gain + second_level_gain +
      (first_level_gain + second_level_gain) * third_level_multiplier

/-- Theorem stating the total number of lives after completing the third level --/
theorem total_lives_after_third_level :
  game_lives 3 = 35 := by sorry

end total_lives_after_third_level_l999_99902


namespace person_a_higher_probability_l999_99994

/-- Represents the space station simulation programming challenge. -/
structure Challenge where
  total_questions : Nat
  questions_per_participant : Nat
  passing_threshold : Nat
  person_a_correct_questions : Nat
  person_b_success_probability : Real

/-- Calculates the probability of passing the challenge given the number of correct programs. -/
def probability_of_passing (c : Challenge) (correct_programs : Nat) : Real :=
  if correct_programs ≥ c.passing_threshold then 1 else 0

/-- Calculates the probability of person B passing the challenge. -/
def person_b_passing_probability (c : Challenge) : Real :=
  sorry

/-- Calculates the probability of person A passing the challenge. -/
def person_a_passing_probability (c : Challenge) : Real :=
  sorry

/-- The main theorem stating that person A has a higher probability of passing the challenge. -/
theorem person_a_higher_probability (c : Challenge) 
  (h1 : c.total_questions = 10)
  (h2 : c.questions_per_participant = 3)
  (h3 : c.passing_threshold = 2)
  (h4 : c.person_a_correct_questions = 6)
  (h5 : c.person_b_success_probability = 0.6) :
  person_a_passing_probability c > person_b_passing_probability c :=
sorry

end person_a_higher_probability_l999_99994


namespace systematic_sampling_theorem_l999_99921

/-- Represents a student number in the range [1, 1000] -/
def StudentNumber := Fin 1000

/-- The total number of students -/
def totalStudents : Nat := 1000

/-- The number of students to be selected in the sample -/
def sampleSize : Nat := 100

/-- The interval between selected students in systematic sampling -/
def samplingInterval : Nat := totalStudents / sampleSize

/-- Predicate to determine if a student number is selected in the systematic sample -/
def isSelected (n : StudentNumber) : Prop :=
  n.val % samplingInterval = 122 % samplingInterval

theorem systematic_sampling_theorem :
  isSelected ⟨121, by norm_num⟩ → isSelected ⟨926, by norm_num⟩ := by
  sorry

end systematic_sampling_theorem_l999_99921


namespace overall_average_percentage_l999_99935

theorem overall_average_percentage (students_A students_B students_C students_D : ℕ)
  (average_A average_B average_C average_D : ℚ) :
  students_A = 15 →
  students_B = 10 →
  students_C = 20 →
  students_D = 5 →
  average_A = 75 / 100 →
  average_B = 90 / 100 →
  average_C = 80 / 100 →
  average_D = 65 / 100 →
  let total_students := students_A + students_B + students_C + students_D
  let total_percentage := students_A * average_A + students_B * average_B +
                          students_C * average_C + students_D * average_D
  total_percentage / total_students = 79 / 100 := by
  sorry

end overall_average_percentage_l999_99935


namespace nonzero_real_solution_cube_equation_l999_99996

theorem nonzero_real_solution_cube_equation (y : ℝ) (h1 : y ≠ 0) (h2 : (3 * y)^5 = (9 * y)^4) : y = 27 := by
  sorry

end nonzero_real_solution_cube_equation_l999_99996


namespace arrangements_count_l999_99907

/-- The number of distinct arrangements of 4 boys and 4 girls in a row,
    where girls cannot be at either end. -/
def num_arrangements : ℕ := 8640

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 4

/-- The total number of people -/
def total_people : ℕ := num_boys + num_girls

/-- Theorem stating that the number of distinct arrangements of 4 boys and 4 girls in a row,
    where girls cannot be at either end, is equal to 8640. -/
theorem arrangements_count :
  num_arrangements = (num_boys * (num_boys - 1)) * Nat.factorial (total_people - 2) :=
by sorry

end arrangements_count_l999_99907


namespace ab_value_l999_99972

theorem ab_value (a b c : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) 
  (h3 : a + c = 2*b) : 
  a * b = 10 := by
sorry

end ab_value_l999_99972


namespace complex_arithmetic_equality_l999_99942

theorem complex_arithmetic_equality : 
  ((-4 : ℝ) ^ 5) ^ (1/5) - (-5 : ℝ) ^ 2 - 5 + ((-43 : ℝ) ^ 4) ^ (1/4) - (-(3 : ℝ) ^ 2) = 0 := by
  sorry

end complex_arithmetic_equality_l999_99942


namespace prize_winning_probability_l999_99904

def num_card_types : ℕ := 3
def num_bags : ℕ := 4

def winning_probability : ℚ :=
  1 - (num_card_types.choose 2 * 2^num_bags - num_card_types) / num_card_types^num_bags

theorem prize_winning_probability :
  winning_probability = 4/9 := by sorry

end prize_winning_probability_l999_99904


namespace yield_contradiction_l999_99947

theorem yield_contradiction (x y z : ℝ) : ¬(0.4 * z + 0.2 * x = 1 ∧
                                           0.1 * y - 0.1 * z = -0.5 ∧
                                           0.1 * x + 0.2 * y = 4) := by
  sorry

end yield_contradiction_l999_99947


namespace option1_cheapest_l999_99976

/-- Regular ticket price -/
def regular_price : ℕ → ℕ := λ x => 40 * x

/-- Platinum card (Option 1) price -/
def platinum_price : ℕ → ℕ := λ x => 200 + 20 * x

/-- Diamond card (Option 2) price -/
def diamond_price : ℕ → ℕ := λ _ => 1000

/-- Theorem: For 8 < x < 40, Option 1 is the cheapest -/
theorem option1_cheapest (x : ℕ) (h1 : 8 < x) (h2 : x < 40) :
  platinum_price x < regular_price x ∧ platinum_price x < diamond_price x :=
by sorry

end option1_cheapest_l999_99976


namespace simplify_expression_l999_99920

theorem simplify_expression (x y : ℝ) (hx : x = 3) (hy : y = 4) : 
  (12 * x^2 * y^3) / (8 * x * y^2) = 18 := by sorry

end simplify_expression_l999_99920


namespace quadratic_equation_solution_l999_99992

theorem quadratic_equation_solution : 
  ∀ a b : ℝ, 
  (∀ x : ℝ, x^2 - 6*x + 13 = 25 ↔ x = a ∨ x = b) → 
  a ≥ b → 
  3*a + 2*b = 15 + Real.sqrt 21 := by
sorry

end quadratic_equation_solution_l999_99992


namespace euler_conjecture_counterexample_l999_99941

theorem euler_conjecture_counterexample : ∃! (n : ℕ), n > 0 ∧ n^5 = 133^5 + 110^5 + 84^5 + 27^5 := by
  sorry

end euler_conjecture_counterexample_l999_99941


namespace point_not_on_line_l999_99990

/-- Given real numbers m and b where mb < 0, the point (1, 2001) does not lie on the line y = m(x^2) + b -/
theorem point_not_on_line (m b : ℝ) (h : m * b < 0) : 
  ¬(2001 = m * (1 : ℝ)^2 + b) := by
  sorry

end point_not_on_line_l999_99990


namespace hemisphere_surface_area_l999_99984

theorem hemisphere_surface_area (r : ℝ) (h1 : r > 0) (h2 : π * r^2 = 225 * π) : 
  3 * π * r^2 + π * r^2 = 900 * π := by
  sorry

end hemisphere_surface_area_l999_99984
