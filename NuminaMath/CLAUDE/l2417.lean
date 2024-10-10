import Mathlib

namespace rose_price_theorem_l2417_241712

/-- The price of an individual rose -/
def individual_rose_price : ℝ := 7.5

/-- The cost of one dozen roses -/
def dozen_price : ℝ := 36

/-- The cost of two dozen roses -/
def two_dozen_price : ℝ := 50

/-- The maximum number of roses that can be purchased for $680 -/
def max_roses : ℕ := 316

/-- The total budget available -/
def total_budget : ℝ := 680

theorem rose_price_theorem :
  (dozen_price = 12 * individual_rose_price) ∧
  (two_dozen_price = 24 * individual_rose_price) ∧
  (∀ n : ℕ, n * individual_rose_price ≤ total_budget → n ≤ max_roses) ∧
  (max_roses * individual_rose_price ≤ total_budget) :=
by sorry

end rose_price_theorem_l2417_241712


namespace no_perfect_square_with_only_six_and_zero_l2417_241722

theorem no_perfect_square_with_only_six_and_zero : 
  ¬ ∃ (n : ℕ), (∃ (m : ℕ), n = m^2) ∧ 
  (∀ (d : ℕ), d ∈ n.digits 10 → d = 6 ∨ d = 0) :=
sorry

end no_perfect_square_with_only_six_and_zero_l2417_241722


namespace sum_representation_l2417_241744

def sum_of_complex_exponentials (z₁ z₂ : ℂ) : ℂ := z₁ + z₂

theorem sum_representation (z₁ z₂ : ℂ) :
  let sum := sum_of_complex_exponentials z₁ z₂
  let r := 30 * Real.cos (π / 10)
  let θ := 9 * π / 20
  z₁ = 15 * Complex.exp (Complex.I * π / 5) ∧
  z₂ = 15 * Complex.exp (Complex.I * 7 * π / 10) →
  sum = r * Complex.exp (Complex.I * θ) :=
by sorry

end sum_representation_l2417_241744


namespace rates_sum_of_squares_l2417_241780

/-- Represents the rates of biking, jogging, and swimming in km/h -/
structure Rates where
  biking : ℕ
  jogging : ℕ
  swimming : ℕ

/-- The sum of the squares of the rates -/
def sumOfSquares (r : Rates) : ℕ :=
  r.biking ^ 2 + r.jogging ^ 2 + r.swimming ^ 2

theorem rates_sum_of_squares : ∃ r : Rates,
  (3 * r.biking + 2 * r.jogging + 2 * r.swimming = 112) ∧
  (2 * r.biking + 3 * r.jogging + 4 * r.swimming = 129) ∧
  (sumOfSquares r = 1218) := by
  sorry

end rates_sum_of_squares_l2417_241780


namespace inequality_theorems_l2417_241713

theorem inequality_theorems :
  (∀ a b : ℝ, a > b → (1 / a < 1 / b → a * b > 0)) ∧
  (∀ a b : ℝ, a > b → (1 / a > 1 / b → a > 0 ∧ 0 > b)) :=
by sorry

end inequality_theorems_l2417_241713


namespace policemen_cover_all_streets_l2417_241704

-- Define the set of intersections
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

-- Define the streets as sets of intersections
def horizontal1 : Set Intersection := {Intersection.A, Intersection.B, Intersection.C, Intersection.D}
def horizontal2 : Set Intersection := {Intersection.E, Intersection.F, Intersection.G}
def horizontal3 : Set Intersection := {Intersection.H, Intersection.I, Intersection.J, Intersection.K}
def vertical1 : Set Intersection := {Intersection.A, Intersection.E, Intersection.H}
def vertical2 : Set Intersection := {Intersection.B, Intersection.F, Intersection.I}
def vertical3 : Set Intersection := {Intersection.D, Intersection.G, Intersection.J}
def diagonal1 : Set Intersection := {Intersection.H, Intersection.F, Intersection.C}
def diagonal2 : Set Intersection := {Intersection.C, Intersection.G, Intersection.K}

-- Define the set of all streets
def allStreets : Set (Set Intersection) := 
  {horizontal1, horizontal2, horizontal3, vertical1, vertical2, vertical3, diagonal1, diagonal2}

-- Define the chosen intersections for policemen
def chosenIntersections : Set Intersection := {Intersection.B, Intersection.G, Intersection.H}

-- Theorem: The chosen intersections cover all streets
theorem policemen_cover_all_streets : 
  ∀ street ∈ allStreets, ∃ intersection ∈ chosenIntersections, intersection ∈ street :=
sorry

end policemen_cover_all_streets_l2417_241704


namespace feeding_theorem_l2417_241717

/-- Represents the number of animal pairs in the sanctuary -/
def num_pairs : ℕ := 6

/-- Represents the feeding order constraint for tigers -/
def tiger_constraint : Prop := true

/-- Represents the constraint that no two same-gender animals can be fed consecutively -/
def alternating_gender_constraint : Prop := true

/-- Represents that the first animal fed is the male lion -/
def starts_with_male_lion : Prop := true

/-- Calculates the number of ways to feed the animals given the constraints -/
def feeding_ways : ℕ := 14400

/-- Theorem stating the number of ways to feed the animals -/
theorem feeding_theorem :
  num_pairs = 6 ∧
  tiger_constraint ∧
  alternating_gender_constraint ∧
  starts_with_male_lion →
  feeding_ways = 14400 := by
  sorry

end feeding_theorem_l2417_241717


namespace x_equals_nine_l2417_241708

theorem x_equals_nine (u : ℤ) (x : ℚ) 
  (h1 : u = -6) 
  (h2 : x = (1 : ℚ) / 3 * (3 - 4 * u)) : 
  x = 9 := by
  sorry

end x_equals_nine_l2417_241708


namespace product_expansion_l2417_241781

theorem product_expansion (x : ℝ) : (2*x + 3) * (3*x^2 + 4*x + 1) = 6*x^3 + 17*x^2 + 14*x + 3 := by
  sorry

end product_expansion_l2417_241781


namespace specific_sculpture_surface_area_l2417_241719

/-- Represents a cube sculpture with a 3x3 bottom layer and a cross-shaped top layer --/
structure CubeSculpture where
  cubeEdgeLength : ℝ
  bottomLayerSize : ℕ
  topLayerSize : ℕ

/-- Calculates the exposed surface area of the cube sculpture --/
def exposedSurfaceArea (sculpture : CubeSculpture) : ℝ :=
  sorry

/-- Theorem stating that the exposed surface area of the specific sculpture is 46 square meters --/
theorem specific_sculpture_surface_area :
  let sculpture : CubeSculpture := {
    cubeEdgeLength := 1,
    bottomLayerSize := 3,
    topLayerSize := 5
  }
  exposedSurfaceArea sculpture = 46 := by sorry

end specific_sculpture_surface_area_l2417_241719


namespace tax_reduction_scientific_notation_l2417_241778

theorem tax_reduction_scientific_notation :
  (15.75 * 10^9 : ℝ) = 1.575 * 10^10 := by sorry

end tax_reduction_scientific_notation_l2417_241778


namespace f_odd_and_decreasing_l2417_241761

def f (x : ℝ) : ℝ := -x^3

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f y < f x) :=
by sorry

end f_odd_and_decreasing_l2417_241761


namespace total_fans_l2417_241711

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- Defines the conditions of the problem -/
def fan_conditions (f : FanCounts) : Prop :=
  f.yankees * 2 = f.mets * 3 ∧  -- Ratio of Yankees to Mets fans is 3:2
  f.mets * 5 = f.red_sox * 4 ∧  -- Ratio of Mets to Red Sox fans is 4:5
  f.mets = 88                   -- There are 88 Mets fans

/-- The theorem to be proved -/
theorem total_fans (f : FanCounts) (h : fan_conditions f) : 
  f.yankees + f.mets + f.red_sox = 330 := by
  sorry

#check total_fans

end total_fans_l2417_241711


namespace palindrome_product_sum_theorem_l2417_241737

/-- A positive three-digit palindrome is a natural number between 100 and 999 (inclusive) 
    that reads the same backwards as forwards. -/
def IsPositiveThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10) ∧ ((n / 10) % 10 = (n % 100) / 10)

/-- The main theorem stating the existence of two positive three-digit palindromes 
    with the given product and sum. -/
theorem palindrome_product_sum_theorem : 
  ∃ (a b : ℕ), IsPositiveThreeDigitPalindrome a ∧ 
                IsPositiveThreeDigitPalindrome b ∧ 
                a * b = 436995 ∧ 
                a + b = 1332 := by
  sorry

end palindrome_product_sum_theorem_l2417_241737


namespace perspective_square_area_l2417_241706

/-- A square whose perspective drawing is a parallelogram with one side of length 4 -/
structure PerspectiveSquare where
  /-- The side length of the original square -/
  side : ℝ
  /-- The side length of the parallelogram in the perspective drawing -/
  perspective_side : ℝ
  /-- The perspective drawing is a parallelogram -/
  is_parallelogram : Bool
  /-- One side of the parallelogram has length 4 -/
  perspective_side_eq_four : perspective_side = 4

/-- The possible areas of the original square -/
def possible_areas (s : PerspectiveSquare) : Set ℝ :=
  {16, 64}

/-- Theorem: The area of the original square is either 16 or 64 -/
theorem perspective_square_area (s : PerspectiveSquare) :
  (s.side ^ 2) ∈ possible_areas s :=
sorry

end perspective_square_area_l2417_241706


namespace domain_intersection_l2417_241787

-- Define the domain of y = √(4-x²)
def domain_sqrt (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

-- Define the domain of y = ln(1-x)
def domain_ln (x : ℝ) : Prop := x < 1

-- Theorem statement
theorem domain_intersection :
  {x : ℝ | domain_sqrt x ∧ domain_ln x} = {x : ℝ | -2 ≤ x ∧ x < 1} :=
sorry

end domain_intersection_l2417_241787


namespace complex_calculations_l2417_241732

theorem complex_calculations :
  let z₁ : ℂ := 1 - 2*I
  let z₂ : ℂ := 3 + 4*I
  let z₃ : ℂ := -2 + I
  let w₁ : ℂ := 1 + 2*I
  let w₂ : ℂ := 3 - 4*I
  (z₁ * z₂ * z₃ = 12 + 9*I) ∧
  (w₁ / w₂ = -1/5 + 2/5*I) := by
sorry

end complex_calculations_l2417_241732


namespace smallest_integer_lower_bound_l2417_241720

theorem smallest_integer_lower_bound 
  (a b c d : ℤ) 
  (different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (average : (a + b + c + d) / 4 = 76) 
  (largest : d = 90) 
  (ordered : a ≤ b ∧ b ≤ c ∧ c ≤ d) : 
  a ≥ 37 := by
  sorry

end smallest_integer_lower_bound_l2417_241720


namespace water_tank_capacity_l2417_241700

theorem water_tank_capacity (c : ℝ) : 
  (c / 3 + 10) / c = 2 / 5 → c = 150 := by
  sorry

end water_tank_capacity_l2417_241700


namespace supermarket_spending_l2417_241718

theorem supermarket_spending (F : ℚ) : 
  F + (1 : ℚ)/3 + (1 : ℚ)/10 + 8/120 = 1 → F = (1 : ℚ)/2 := by
  sorry

end supermarket_spending_l2417_241718


namespace impossibility_of_transformation_l2417_241768

def operation (a b : ℤ) : ℤ × ℤ := (5*a - 2*b, 3*a - 4*b)

def initial_set : Set ℤ := {n | 1 ≤ n ∧ n ≤ 2018}

def target_sequence : Set ℤ := {n | ∃ k, 1 ≤ k ∧ k ≤ 2018 ∧ n = 3*k}

theorem impossibility_of_transformation :
  ∀ (S : Set ℤ), S = initial_set →
  ¬∃ (n : ℕ), ∃ (S' : Set ℤ),
    (∀ k ≤ n, ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧
      S' = (S \ {a, b}) ∪ {(operation a b).1, (operation a b).2}) →
    target_sequence ⊆ S' :=
sorry

end impossibility_of_transformation_l2417_241768


namespace special_ellipse_properties_l2417_241705

/-- An ellipse with a vertex at (0,1) and focal length 2√3 -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h0 : a > b
  h1 : b > 0
  h2 : b = 1
  h3 : a^2 = b^2 + 3

/-- The intersection points of a line with the ellipse -/
def LineEllipseIntersection (E : SpecialEllipse) (k : ℝ) :=
  {x : ℝ × ℝ | ∃ t, x.1 = -2 + t ∧ x.2 = 1 + k*t ∧ (x.1^2 / E.a^2 + x.2^2 / E.b^2 = 1)}

/-- The x-intercepts of lines connecting (0,1) to the intersection points -/
def XIntercepts (E : SpecialEllipse) (k : ℝ) (B C : ℝ × ℝ) :=
  {x : ℝ | ∃ t, (t*B.1 = x ∧ t*B.2 = 1) ∨ (t*C.1 = x ∧ t*C.2 = 1)}

theorem special_ellipse_properties (E : SpecialEllipse) :
  (∀ x y, x^2/4 + y^2 = 1 ↔ x^2/E.a^2 + y^2/E.b^2 = 1) ∧
  (∀ k : ℝ, k ≠ 0 →
    ∀ B C : ℝ × ℝ, B ∈ LineEllipseIntersection E k → C ∈ LineEllipseIntersection E k → B ≠ C →
    ∀ M N : ℝ, M ∈ XIntercepts E k B C → N ∈ XIntercepts E k B C → M ≠ N →
    (M - N)^2 * |k| = 16) :=
sorry

end special_ellipse_properties_l2417_241705


namespace equation_solution_l2417_241772

theorem equation_solution : ∃! x : ℝ, (x ≠ 1 ∧ x ≠ -1) ∧ (x / (x - 1) = 2 / (x^2 - 1)) ∧ x = -2 := by
  sorry

end equation_solution_l2417_241772


namespace polynomial_difference_divisibility_l2417_241789

/-- For any polynomial P with integer coefficients and any integers a and b,
    (a - b) divides (P(a) - P(b)) in ℤ. -/
theorem polynomial_difference_divisibility (P : Polynomial ℤ) (a b : ℤ) :
  (a - b) ∣ (P.eval a - P.eval b) :=
sorry

end polynomial_difference_divisibility_l2417_241789


namespace road_repaving_l2417_241790

theorem road_repaving (total_repaved : ℕ) (repaved_today : ℕ) 
  (h1 : total_repaved = 4938)
  (h2 : repaved_today = 805) :
  total_repaved - repaved_today = 4133 :=
by
  sorry

end road_repaving_l2417_241790


namespace average_stickers_per_album_l2417_241738

def album_stickers : List ℕ := [5, 7, 9, 14, 19, 12, 26, 18, 11, 15]

theorem average_stickers_per_album :
  (album_stickers.sum : ℚ) / album_stickers.length = 13.6 := by
  sorry

end average_stickers_per_album_l2417_241738


namespace monomial_exponents_sum_l2417_241729

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (a b : ℕ) (c d : ℤ) : Prop :=
  a = 3 ∧ b = 1 ∧ c = 3 ∧ d = 1

theorem monomial_exponents_sum (m n : ℕ) :
  like_terms m 1 3 n → m + n = 4 := by
  sorry

end monomial_exponents_sum_l2417_241729


namespace abrahams_budget_l2417_241775

/-- Abraham's shopping budget problem -/
theorem abrahams_budget :
  let shower_gel_count : ℕ := 4
  let shower_gel_price : ℕ := 4
  let toothpaste_price : ℕ := 3
  let laundry_detergent_price : ℕ := 11
  let remaining_budget : ℕ := 30
  shower_gel_count * shower_gel_price + toothpaste_price + laundry_detergent_price + remaining_budget = 60
  := by sorry

end abrahams_budget_l2417_241775


namespace smaller_cubes_count_l2417_241764

theorem smaller_cubes_count (larger_cube_volume : ℝ) (smaller_cube_volume : ℝ) (surface_area_difference : ℝ) :
  larger_cube_volume = 216 →
  smaller_cube_volume = 1 →
  surface_area_difference = 1080 →
  (smaller_cube_volume^(1/3) * 6 * (larger_cube_volume / smaller_cube_volume) - larger_cube_volume^(2/3) * 6 = surface_area_difference) →
  (larger_cube_volume / smaller_cube_volume) = 216 :=
by
  sorry

#check smaller_cubes_count

end smaller_cubes_count_l2417_241764


namespace greatest_product_sum_300_l2417_241726

theorem greatest_product_sum_300 : 
  ∀ x y : ℤ, x + y = 300 → x * y ≤ 22500 := by
  sorry

end greatest_product_sum_300_l2417_241726


namespace fuel_for_three_trips_l2417_241788

/-- Calculates the total fuel needed for a series of trips given a fuel consumption rate -/
def totalFuelNeeded (fuelRate : ℝ) (trips : List ℝ) : ℝ :=
  fuelRate * (trips.sum)

/-- Proves that the total fuel needed for three specific trips is 550 liters -/
theorem fuel_for_three_trips :
  let fuelRate : ℝ := 5
  let trips : List ℝ := [50, 35, 25]
  totalFuelNeeded fuelRate trips = 550 := by
  sorry

#check fuel_for_three_trips

end fuel_for_three_trips_l2417_241788


namespace coprime_divides_l2417_241776

theorem coprime_divides (a b n : ℕ) : 
  Nat.Coprime a b → a ∣ n → b ∣ n → (a * b) ∣ n := by
  sorry

end coprime_divides_l2417_241776


namespace product_of_primes_l2417_241797

theorem product_of_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p * q = 69 →
  13 < q →
  q < 25 →
  15 < p * q →
  p * q < 70 →
  p = 3 := by
sorry

end product_of_primes_l2417_241797


namespace quadratic_through_origin_l2417_241750

/-- A quadratic function passing through the origin -/
def passes_through_origin (a b c : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 + b * x + c = 0

theorem quadratic_through_origin (a b c : ℝ) (h : a ≠ 0) :
  passes_through_origin a b c ↔ c = 0 := by sorry

end quadratic_through_origin_l2417_241750


namespace tangent_line_intersection_l2417_241785

-- Define the circles
def circle1 : ℝ × ℝ := (0, 0)
def circle2 : ℝ × ℝ := (20, 0)
def radius1 : ℝ := 3
def radius2 : ℝ := 9

-- Define the tangent line intersection point
def intersection_point : ℝ := 5

-- Theorem statement
theorem tangent_line_intersection :
  let d := circle2.1 - circle1.1  -- Distance between circle centers
  ∃ (t : ℝ), 
    t > 0 ∧ 
    intersection_point = circle1.1 + t * radius1 ∧
    intersection_point = circle2.1 - t * radius2 ∧
    t * (radius1 + radius2) = d :=
sorry

end tangent_line_intersection_l2417_241785


namespace tripled_room_painting_cost_l2417_241733

/-- Represents the cost of painting a room -/
structure PaintingCost where
  original : ℝ
  scaled : ℝ

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  breadth : ℝ
  height : ℝ

/-- Calculates the wall area of a room given its dimensions -/
def wallArea (d : RoomDimensions) : ℝ :=
  2 * (d.length + d.breadth) * d.height

/-- Scales the dimensions of a room by a factor -/
def scaleDimensions (d : RoomDimensions) (factor : ℝ) : RoomDimensions :=
  { length := d.length * factor
  , breadth := d.breadth * factor
  , height := d.height * factor }

/-- Theorem: The cost of painting a room with tripled dimensions is Rs. 3150 
    given that the original cost is Rs. 350 -/
theorem tripled_room_painting_cost 
  (d : RoomDimensions) 
  (c : PaintingCost) 
  (h1 : c.original = 350) 
  (h2 : c.original / wallArea d = c.scaled / wallArea (scaleDimensions d 3)) : 
  c.scaled = 3150 := by
  sorry

end tripled_room_painting_cost_l2417_241733


namespace hyperbola_equation_l2417_241799

-- Define the hyperbola
structure Hyperbola where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  passes_through : ℝ × ℝ  -- point that the hyperbola passes through

-- Define the standard equation of a hyperbola
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

-- Theorem statement
theorem hyperbola_equation (h : Hyperbola) :
  h.a = 2 * Real.sqrt 5 →
  h.passes_through = (5, -2) →
  ∀ x y : ℝ, standard_equation h x y ↔ x^2 / 20 - y^2 / 16 = 1 :=
sorry

end hyperbola_equation_l2417_241799


namespace unique_solution_exists_l2417_241795

-- Define the system of equations
def system (a b c x y z : ℝ) : Prop :=
  Real.sqrt (y - a) + Real.sqrt (z - a) = 1 ∧
  Real.sqrt (z - b) + Real.sqrt (x - b) = 1 ∧
  Real.sqrt (x - c) + Real.sqrt (y - c) = 1

-- Theorem statement
theorem unique_solution_exists (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : Real.sqrt a + Real.sqrt b + Real.sqrt c = Real.sqrt 3 / 2) :
  ∃! x y z : ℝ, system a b c x y z :=
sorry

end unique_solution_exists_l2417_241795


namespace handshake_problem_l2417_241763

theorem handshake_problem :
  let n : ℕ := 6  -- number of people
  let handshakes := n * (n - 1) / 2  -- formula for total handshakes
  handshakes = 15
  := by sorry

end handshake_problem_l2417_241763


namespace dropped_students_scores_sum_l2417_241709

theorem dropped_students_scores_sum 
  (initial_students : ℕ) 
  (initial_average : ℝ) 
  (remaining_students : ℕ) 
  (new_average : ℝ) 
  (h1 : initial_students = 25) 
  (h2 : initial_average = 60.5) 
  (h3 : remaining_students = 23) 
  (h4 : new_average = 64.0) : 
  (initial_students : ℝ) * initial_average - (remaining_students : ℝ) * new_average = 40.5 := by
  sorry

end dropped_students_scores_sum_l2417_241709


namespace distance_to_origin_l2417_241766

theorem distance_to_origin (x y : ℝ) (h1 : y = 15) (h2 : x = 2 + Real.sqrt 105)
  (h3 : Real.sqrt ((x - 2)^2 + (y - 7)^2) = 13) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (334 + 4 * Real.sqrt 105) := by
  sorry

end distance_to_origin_l2417_241766


namespace beach_creatures_ratio_l2417_241716

theorem beach_creatures_ratio :
  ∀ (oysters_day1 crabs_day1 total_both_days : ℕ),
    oysters_day1 = 50 →
    crabs_day1 = 72 →
    total_both_days = 195 →
    ∃ (crabs_day2 : ℕ),
      oysters_day1 + crabs_day1 + (oysters_day1 / 2 + crabs_day2) = total_both_days ∧
      crabs_day2 * 3 = crabs_day1 * 2 :=
by sorry

end beach_creatures_ratio_l2417_241716


namespace relay_race_distance_ratio_l2417_241723

theorem relay_race_distance_ratio :
  ∀ (last_year_distance : ℕ) (table_count : ℕ) (distance_1_to_3 : ℕ),
    last_year_distance = 300 →
    table_count = 6 →
    distance_1_to_3 = 400 →
    ∃ (this_year_distance : ℕ),
      this_year_distance % last_year_distance = 0 ∧
      (this_year_distance : ℚ) / last_year_distance = 10 / 3 := by
  sorry

end relay_race_distance_ratio_l2417_241723


namespace logarithm_inequality_l2417_241710

theorem logarithm_inequality (m : ℝ) (a b c : ℝ) 
  (h1 : 1/10 < m ∧ m < 1) 
  (h2 : a = Real.log m) 
  (h3 : b = Real.log (m^2)) 
  (h4 : c = Real.log (m^3)) : 
  b < a ∧ a < c := by
  sorry

end logarithm_inequality_l2417_241710


namespace field_trip_difference_l2417_241758

/-- Given the number of vans, buses, people per van, and people per bus,
    prove that the difference between the number of people traveling by bus
    and the number of people traveling by van is 108.0 --/
theorem field_trip_difference (num_vans : ℝ) (num_buses : ℝ) 
                               (people_per_van : ℝ) (people_per_bus : ℝ) :
  num_vans = 6.0 →
  num_buses = 8.0 →
  people_per_van = 6.0 →
  people_per_bus = 18.0 →
  num_buses * people_per_bus - num_vans * people_per_van = 108.0 :=
by sorry

end field_trip_difference_l2417_241758


namespace coefficient_x6_in_x_plus_2_to_8_l2417_241742

theorem coefficient_x6_in_x_plus_2_to_8 :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k * 2^(8 - k)) * (if k = 6 then 1 else 0)) = 112 := by
  sorry

end coefficient_x6_in_x_plus_2_to_8_l2417_241742


namespace min_equation_implies_sum_l2417_241798

theorem min_equation_implies_sum (a b c d : ℝ) :
  (∀ x : ℝ, min (20 * x + 19) (19 * x + 20) = (a * x + b) - |c * x + d|) →
  a * b + c * d = 380 := by
  sorry

end min_equation_implies_sum_l2417_241798


namespace six_digit_square_from_three_squares_l2417_241777

/-- A function that concatenates three two-digit numbers into a six-digit number -/
def concatenate (a b c : Nat) : Nat :=
  10000 * a + 100 * b + c

/-- A predicate that checks if a number is a two-digit perfect square -/
def is_two_digit_perfect_square (n : Nat) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ k, k * k = n

/-- The main theorem statement -/
theorem six_digit_square_from_three_squares :
  ∀ a b c : Nat,
    is_two_digit_perfect_square a →
    is_two_digit_perfect_square b →
    is_two_digit_perfect_square c →
    (∃ t : Nat, t * t = concatenate a b c) →
    concatenate a b c = 166464 ∨ concatenate a b c = 646416 := by
  sorry


end six_digit_square_from_three_squares_l2417_241777


namespace trihedral_angle_sum_bounds_l2417_241730

-- Define a trihedral angle
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real

-- State the theorem
theorem trihedral_angle_sum_bounds (t : TrihedralAngle) :
  180 < t.α + t.β + t.γ ∧ t.α + t.β + t.γ < 540 := by
  sorry

end trihedral_angle_sum_bounds_l2417_241730


namespace square_minus_product_plus_square_l2417_241755

theorem square_minus_product_plus_square : 5^2 - 3*4 + 3^2 = 22 := by
  sorry

end square_minus_product_plus_square_l2417_241755


namespace lead_is_seventeen_l2417_241791

-- Define the scores of both teams
def chucks_team_score : ℕ := 72
def yellow_team_score : ℕ := 55

-- Define the lead as the difference between the scores
def lead : ℕ := chucks_team_score - yellow_team_score

-- Theorem stating that the lead is 17 points
theorem lead_is_seventeen : lead = 17 := by
  sorry

end lead_is_seventeen_l2417_241791


namespace triangle_theorem_l2417_241784

noncomputable def triangle_proof (A B C : ℝ) (a b c : ℝ) : Prop :=
  let perimeter := a + b + c
  let area := (1/2) * b * c * Real.sin A
  perimeter = 4 * (Real.sqrt 2 + 1) ∧
  Real.sin B + Real.sin C = Real.sqrt 2 * Real.sin A ∧
  area = 3 * Real.sin A ∧
  a = 4 ∧
  A = Real.arccos (1/3)

theorem triangle_theorem :
  ∀ (A B C : ℝ) (a b c : ℝ),
  triangle_proof A B C a b c :=
by
  sorry

end triangle_theorem_l2417_241784


namespace average_minutes_run_per_day_l2417_241753

/-- The average number of minutes run per day by sixth graders -/
def sixth_grade_avg : ℚ := 20

/-- The average number of minutes run per day by seventh graders -/
def seventh_grade_avg : ℚ := 12

/-- The average number of minutes run per day by eighth graders -/
def eighth_grade_avg : ℚ := 18

/-- The ratio of sixth graders to eighth graders -/
def sixth_to_eighth_ratio : ℚ := 3

/-- The ratio of seventh graders to eighth graders -/
def seventh_to_eighth_ratio : ℚ := 3

/-- The theorem stating the average number of minutes run per day by all students -/
theorem average_minutes_run_per_day :
  let total_students := sixth_to_eighth_ratio + seventh_to_eighth_ratio + 1
  let total_minutes := sixth_grade_avg * sixth_to_eighth_ratio + 
                       seventh_grade_avg * seventh_to_eighth_ratio + 
                       eighth_grade_avg
  total_minutes / total_students = 114 / 7 := by
  sorry

end average_minutes_run_per_day_l2417_241753


namespace shaded_area_outside_overlap_l2417_241751

/-- Given two rectangles with specific dimensions and overlap, calculate the shaded area outside the overlap -/
theorem shaded_area_outside_overlap (rect1_width rect1_height rect2_width rect2_height overlap_width overlap_height : ℕ) 
  (h1 : rect1_width = 4 ∧ rect1_height = 12)
  (h2 : rect2_width = 5 ∧ rect2_height = 9)
  (h3 : overlap_width = 4 ∧ overlap_height = 5) :
  rect1_width * rect1_height + rect2_width * rect2_height - overlap_width * overlap_height = 73 :=
by sorry

end shaded_area_outside_overlap_l2417_241751


namespace not_always_possible_to_make_all_white_l2417_241749

/-- Represents a smaller equilateral triangle within the larger triangle -/
structure SmallTriangle where
  color : Bool  -- true for white, false for black

/-- Represents the entire configuration of the divided equilateral triangle -/
structure TriangleConfiguration where
  smallTriangles : List SmallTriangle
  numRows : Nat  -- number of rows in the triangle

/-- Represents a repainting operation -/
def repaint (config : TriangleConfiguration) (lineIndex : Nat) : TriangleConfiguration :=
  sorry

/-- Checks if all small triangles in the configuration are white -/
def allWhite (config : TriangleConfiguration) : Bool :=
  sorry

/-- Theorem stating that there exists a configuration where it's impossible to make all triangles white -/
theorem not_always_possible_to_make_all_white :
  ∃ (initialConfig : TriangleConfiguration),
    ∀ (repaintSequence : List Nat),
      let finalConfig := repaintSequence.foldl repaint initialConfig
      ¬(allWhite finalConfig) :=
sorry

end not_always_possible_to_make_all_white_l2417_241749


namespace sum_integers_neg25_to_55_l2417_241767

/-- The sum of integers from a to b, inclusive -/
def sum_integers (a b : ℤ) : ℤ := (b - a + 1) * (a + b) / 2

/-- Theorem: The sum of integers from -25 to 55 is 1215 -/
theorem sum_integers_neg25_to_55 : sum_integers (-25) 55 = 1215 := by
  sorry

end sum_integers_neg25_to_55_l2417_241767


namespace uki_earnings_l2417_241747

/-- Represents Uki's bakery business -/
structure BakeryBusiness where
  cupcake_price : ℝ
  cookie_price : ℝ
  biscuit_price : ℝ
  daily_cupcakes : ℕ
  daily_cookies : ℕ
  daily_biscuits : ℕ

/-- Calculates the total earnings for a given number of days -/
def total_earnings (b : BakeryBusiness) (days : ℕ) : ℝ :=
  (b.cupcake_price * b.daily_cupcakes + 
   b.cookie_price * b.daily_cookies + 
   b.biscuit_price * b.daily_biscuits) * days

/-- Theorem stating that Uki's total earnings for five days is $350 -/
theorem uki_earnings : ∃ (b : BakeryBusiness), 
  b.cupcake_price = 1.5 ∧ 
  b.cookie_price = 2 ∧ 
  b.biscuit_price = 1 ∧ 
  b.daily_cupcakes = 20 ∧ 
  b.daily_cookies = 10 ∧ 
  b.daily_biscuits = 20 ∧ 
  total_earnings b 5 = 350 := by
  sorry

end uki_earnings_l2417_241747


namespace square_area_from_perimeter_l2417_241754

theorem square_area_from_perimeter (perimeter : ℝ) (h : perimeter = 52) :
  (perimeter / 4) ^ 2 = 169 := by
  sorry

end square_area_from_perimeter_l2417_241754


namespace inequality_equivalence_l2417_241779

theorem inequality_equivalence (x : ℝ) : 
  |((7-x)/4)| < 3 ↔ 2 < x ∧ x < 19 := by
  sorry

end inequality_equivalence_l2417_241779


namespace lacrosse_football_difference_l2417_241731

/-- Represents the number of bottles filled for each team and the total --/
structure BottleCounts where
  total : ℕ
  football : ℕ
  soccer : ℕ
  rugby : ℕ
  lacrosse : ℕ

/-- The difference in bottles between lacrosse and football teams --/
def bottleDifference (counts : BottleCounts) : ℕ :=
  counts.lacrosse - counts.football

/-- Theorem stating the difference in bottles between lacrosse and football teams --/
theorem lacrosse_football_difference (counts : BottleCounts) 
  (h1 : counts.total = 254)
  (h2 : counts.football = 11 * 6)
  (h3 : counts.soccer = 53)
  (h4 : counts.rugby = 49)
  (h5 : counts.total = counts.football + counts.soccer + counts.rugby + counts.lacrosse) :
  bottleDifference counts = 20 := by
  sorry

#check lacrosse_football_difference

end lacrosse_football_difference_l2417_241731


namespace half_circle_roll_center_path_length_l2417_241793

/-- The length of the path traveled by the center of a half-circle when rolled along a straight line -/
def half_circle_center_path_length (radius : ℝ) : ℝ := 2 * radius

/-- Theorem: The length of the path traveled by the center of a half-circle with radius 1 cm, 
    when rolled along a straight line until it completes a half rotation, is equal to 2 cm -/
theorem half_circle_roll_center_path_length :
  half_circle_center_path_length 1 = 2 := by sorry

end half_circle_roll_center_path_length_l2417_241793


namespace first_quarter_time_proportion_l2417_241736

/-- Represents the proportion of time spent traveling the first quarter of a distance
    when the speed for that quarter is 4 times the speed for the remaining distance -/
def time_proportion_first_quarter : ℚ := 1 / 13

/-- Proves that the proportion of time spent traveling the first quarter of the distance
    is 1/13 of the total time, given the specified speed conditions -/
theorem first_quarter_time_proportion 
  (D : ℝ) -- Total distance
  (V : ℝ) -- Speed for the remaining three-quarters of the distance
  (h1 : D > 0) -- Distance is positive
  (h2 : V > 0) -- Speed is positive
  : (D / (16 * V)) / ((D / (16 * V)) + (3 * D / (4 * V))) = time_proportion_first_quarter :=
sorry

end first_quarter_time_proportion_l2417_241736


namespace unknown_blanket_rate_solve_unknown_blanket_rate_l2417_241725

/-- Proves that the unknown rate of two blankets is 275, given the conditions of the problem --/
theorem unknown_blanket_rate : ℕ → Prop := fun x =>
  let total_blankets : ℕ := 12
  let average_price : ℕ := 150
  let total_cost : ℕ := total_blankets * average_price
  let known_cost : ℕ := 5 * 100 + 5 * 150
  2 * x = total_cost - known_cost → x = 275

/-- Solution to the unknown_blanket_rate theorem --/
theorem solve_unknown_blanket_rate : unknown_blanket_rate 275 := by
  sorry

end unknown_blanket_rate_solve_unknown_blanket_rate_l2417_241725


namespace trigonometric_identity_l2417_241757

theorem trigonometric_identity (α β γ n : ℝ) 
  (h : Real.sin (2 * (α + γ)) = n * Real.sin (2 * β)) :
  Real.tan (α + β + γ) / Real.tan (α - β + γ) = (n + 1) / (n - 1) := by
  sorry

end trigonometric_identity_l2417_241757


namespace purely_imaginary_implies_m_is_one_l2417_241792

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Given that z = (m^2 - m) + mi is purely imaginary and m is real, prove that m = 1. -/
theorem purely_imaginary_implies_m_is_one (m : ℝ) :
  isPurelyImaginary ((m^2 - m : ℝ) + m * I) → m = 1 := by
  sorry


end purely_imaginary_implies_m_is_one_l2417_241792


namespace solve_equation_l2417_241739

theorem solve_equation (C D : ℚ) 
  (eq1 : 2 * C + 3 * D + 4 = 31)
  (eq2 : D = C + 2) :
  C = 21 / 5 ∧ D = 31 / 5 := by
sorry

end solve_equation_l2417_241739


namespace sum_of_456_l2417_241724

/-- A geometric sequence with first term 3 and sum of first three terms 9 -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 2) * a n = (a (n + 1))^2
  first_term : a 1 = 3
  sum_first_three : a 1 + a 2 + a 3 = 9

/-- The sum of the 4th, 5th, and 6th terms is either 9 or -72 -/
theorem sum_of_456 (seq : GeometricSequence) :
  seq.a 4 + seq.a 5 + seq.a 6 = 9 ∨ seq.a 4 + seq.a 5 + seq.a 6 = -72 := by
  sorry

end sum_of_456_l2417_241724


namespace larger_number_four_times_smaller_l2417_241771

theorem larger_number_four_times_smaller
  (a b : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_distinct : a ≠ b)
  (h_equation : a^3 - b^3 = 3*(2*a^2*b - 3*a*b^2 + b^3)) :
  a = 4*b :=
sorry

end larger_number_four_times_smaller_l2417_241771


namespace expression_evaluation_l2417_241714

theorem expression_evaluation :
  let x := Real.sqrt 2 * Real.sin (π / 4) + Real.tan (π / 3)
  (x / (x^2 - 1)) / (1 - 1 / (x + 1)) = Real.sqrt 3 / 3 := by
  sorry

end expression_evaluation_l2417_241714


namespace smallest_4_9_divisible_by_4_and_9_l2417_241774

def is_composed_of_4_and_9 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 4 ∨ d = 9

theorem smallest_4_9_divisible_by_4_and_9 :
  ∃! n : ℕ,
    n > 0 ∧
    n % 4 = 0 ∧
    n % 9 = 0 ∧
    is_composed_of_4_and_9 n ∧
    ∀ m : ℕ, m > 0 ∧ m % 4 = 0 ∧ m % 9 = 0 ∧ is_composed_of_4_and_9 m → n ≤ m :=
by
  sorry

end smallest_4_9_divisible_by_4_and_9_l2417_241774


namespace evelyn_marbles_count_l2417_241752

def initial_marbles : ℕ := 95
def marbles_from_henry : ℕ := 9
def cards_bought : ℕ := 6

theorem evelyn_marbles_count :
  initial_marbles + marbles_from_henry = 104 :=
by sorry

end evelyn_marbles_count_l2417_241752


namespace intersection_points_l2417_241770

/-- The number of intersection points for k lines in a plane -/
def f (k : ℕ) : ℕ := sorry

/-- No two lines are parallel and no three lines intersect at the same point -/
axiom line_properties (k : ℕ) : True

theorem intersection_points (k : ℕ) : f (k + 1) = f k + k :=
  sorry

end intersection_points_l2417_241770


namespace our_ellipse_correct_l2417_241796

/-- An ellipse with foci at (-2, 0) and (2, 0) passing through (2, 3) -/
structure Ellipse where
  -- The equation of the ellipse
  equation : ℝ → ℝ → Prop
  -- The foci are at (-2, 0) and (2, 0)
  foci_x : equation (-2) 0 ∧ equation 2 0
  -- The ellipse passes through (2, 3)
  passes_through : equation 2 3
  -- The equation is of the form x^2/a^2 + y^2/b^2 = 1 for some a, b
  is_standard_form : ∃ a b : ℝ, ∀ x y : ℝ, equation x y ↔ x^2/a^2 + y^2/b^2 = 1

/-- The specific ellipse we're interested in -/
def our_ellipse : Ellipse where
  equation := fun x y => x^2/16 + y^2/12 = 1
  foci_x := sorry
  passes_through := sorry
  is_standard_form := sorry

/-- The theorem stating that our_ellipse satisfies all the conditions -/
theorem our_ellipse_correct : 
  our_ellipse.equation = fun x y => x^2/16 + y^2/12 = 1 := by sorry

end our_ellipse_correct_l2417_241796


namespace least_n_for_inequality_l2417_241746

theorem least_n_for_inequality : ∃ n : ℕ, n > 0 ∧ 
  (∀ k : ℕ, k > 0 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) < (1 : ℚ) / 8 → k ≥ n) ∧
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 8) ∧ n = 3 :=
by sorry

end least_n_for_inequality_l2417_241746


namespace triangle_min_value_l2417_241721

theorem triangle_min_value (a b c : ℝ) (A : ℝ) (area : ℝ) : 
  A = Real.pi / 3 →
  area = Real.sqrt 3 →
  area = (1 / 2) * b * c * Real.sin A →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  (∀ x y : ℝ, (4 * x ^ 2 + 4 * y ^ 2 - 3 * a ^ 2) / (x + y) ≥ 5) ∧
  (∃ x y : ℝ, (4 * x ^ 2 + 4 * y ^ 2 - 3 * a ^ 2) / (x + y) = 5) :=
by sorry

end triangle_min_value_l2417_241721


namespace ab_length_in_specific_triangle_l2417_241728

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def isAcute (t : Triangle) : Prop := sorry

def sideLength (p1 p2 : ℝ × ℝ) : ℝ := sorry

def triangleArea (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem ab_length_in_specific_triangle :
  ∀ (t : Triangle),
    isAcute t →
    sideLength t.A t.C = 4 →
    sideLength t.B t.C = 3 →
    triangleArea t = 3 * Real.sqrt 3 →
    sideLength t.A t.B = Real.sqrt 13 := by
  sorry


end ab_length_in_specific_triangle_l2417_241728


namespace shared_fixed_points_l2417_241783

/-- A function that represents f(x) = x^2 - 2 --/
def f (x : ℝ) : ℝ := x^2 - 2

/-- A function that represents g(x) = 2x^2 - c --/
def g (c : ℝ) (x : ℝ) : ℝ := 2*x^2 - c

/-- The theorem stating the conditions for shared fixed points --/
theorem shared_fixed_points (c : ℝ) : 
  (c = 3 ∨ c = 6) ↔ ∃ x : ℝ, (f x = x ∧ g c x = x) :=
sorry

end shared_fixed_points_l2417_241783


namespace lowest_number_of_students_twenty_four_is_lowest_l2417_241782

theorem lowest_number_of_students (n : ℕ) : n > 0 ∧ 12 ∣ n ∧ 24 ∣ n → n ≥ 24 := by
  sorry

theorem twenty_four_is_lowest : ∃ (n : ℕ), n > 0 ∧ 12 ∣ n ∧ 24 ∣ n ∧ n = 24 := by
  sorry

end lowest_number_of_students_twenty_four_is_lowest_l2417_241782


namespace elf_circle_arrangement_exists_l2417_241707

/-- Represents the height of an elf -/
inductive ElfHeight
| Short
| Tall

/-- Represents an elf in the circle -/
structure Elf :=
  (position : Nat)
  (height : ElfHeight)

/-- Checks if an elf is taller than both neighbors -/
def isTallerThanNeighbors (elves : List Elf) (position : Nat) : Bool :=
  sorry

/-- Checks if an elf is shorter than both neighbors -/
def isShorterThanNeighbors (elves : List Elf) (position : Nat) : Bool :=
  sorry

/-- Checks if all elves in the circle satisfy the eye-closing condition -/
def allElvesSatisfyCondition (elves : List Elf) : Bool :=
  sorry

/-- Theorem: There exists an arrangement of 100 elves that satisfies all conditions -/
theorem elf_circle_arrangement_exists : 
  ∃ (elves : List Elf), 
    elves.length = 100 ∧ 
    (∀ e ∈ elves, e.position ≤ 100) ∧
    allElvesSatisfyCondition elves :=
  sorry

end elf_circle_arrangement_exists_l2417_241707


namespace sqrt_equation_solution_l2417_241715

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 15) = 12 → x = 129 := by
  sorry

end sqrt_equation_solution_l2417_241715


namespace reflection_y_axis_l2417_241759

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

theorem reflection_y_axis : 
  let A : ℝ × ℝ := (-3, 4)
  reflect_y A = (3, 4) := by sorry

end reflection_y_axis_l2417_241759


namespace donut_selections_l2417_241741

theorem donut_selections :
  (Nat.choose 9 3) = 84 := by
  sorry

end donut_selections_l2417_241741


namespace tournament_committee_count_l2417_241786

/-- Represents a frisbee league -/
structure FrisbeeLeague where
  teams : Nat
  membersPerTeam : Nat
  committeeSize : Nat
  hostTeamMembers : Nat
  nonHostTeamMembers : Nat

/-- The specific frisbee league described in the problem -/
def regionalLeague : FrisbeeLeague :=
  { teams := 5
  , membersPerTeam := 8
  , committeeSize := 11
  , hostTeamMembers := 4
  , nonHostTeamMembers := 3 }

/-- The number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  Nat.choose n k

/-- The number of possible tournament committees -/
def numberOfCommittees (league : FrisbeeLeague) : Nat :=
  league.teams *
  (choose (league.membersPerTeam - 1) (league.hostTeamMembers - 1)) *
  (choose league.membersPerTeam league.nonHostTeamMembers ^ (league.teams - 1))

/-- Theorem stating the number of possible tournament committees -/
theorem tournament_committee_count :
  numberOfCommittees regionalLeague = 1723286800 := by
  sorry

end tournament_committee_count_l2417_241786


namespace largest_multiple_eleven_l2417_241745

theorem largest_multiple_eleven (n : ℤ) : 
  (n * 11 = -209) → 
  (-n * 11 > -210) ∧ 
  ∀ m : ℤ, (m > n) → (-m * 11 ≤ -210) :=
by
  sorry

end largest_multiple_eleven_l2417_241745


namespace coin_value_increase_l2417_241703

def coins_bought : ℕ := 20
def initial_price : ℚ := 15
def coins_sold : ℕ := 12

def original_investment : ℚ := coins_bought * initial_price
def selling_price : ℚ := original_investment

theorem coin_value_increase :
  (selling_price / coins_sold - initial_price) / initial_price = 2/3 :=
sorry

end coin_value_increase_l2417_241703


namespace debt_installment_problem_l2417_241773

/-- Proves that given 52 installments where the first 12 are x and the remaining 40 are (x + 65),
    if the average payment is $460, then x = $410. -/
theorem debt_installment_problem (x : ℝ) : 
  (12 * x + 40 * (x + 65)) / 52 = 460 → x = 410 := by
  sorry

end debt_installment_problem_l2417_241773


namespace cube_sum_fraction_l2417_241794

theorem cube_sum_fraction (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 8 = 219/8 := by
  sorry

end cube_sum_fraction_l2417_241794


namespace seven_power_minus_three_times_two_power_l2417_241740

theorem seven_power_minus_three_times_two_power (x y : ℕ+) : 
  7^(x.val) - 3 * 2^(y.val) = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) := by
sorry

end seven_power_minus_three_times_two_power_l2417_241740


namespace min_value_of_sum_of_squares_l2417_241769

theorem min_value_of_sum_of_squares (x y : ℝ) : 
  (x + 5)^2 + (y - 12)^2 = 14^2 → ∃ (min : ℝ), min = 1 ∧ ∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → x^2 + y^2 ≤ a^2 + b^2 := by
  sorry

#check min_value_of_sum_of_squares

end min_value_of_sum_of_squares_l2417_241769


namespace water_amount_in_sport_formulation_l2417_241735

/-- Represents the ratios in a drink formulation -/
structure DrinkRatio where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation of the drink -/
def standard_ratio : DrinkRatio := ⟨1, 12, 30⟩

/-- The sport formulation of the drink -/
def sport_ratio : DrinkRatio :=
  ⟨1, 
   3 * standard_ratio.corn_syrup / standard_ratio.flavoring,
   standard_ratio.water / (2 * standard_ratio.flavoring)⟩

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def corn_syrup_amount : ℚ := 8

/-- Theorem stating the amount of water in the sport formulation -/
theorem water_amount_in_sport_formulation :
  (corn_syrup_amount * sport_ratio.water) / sport_ratio.corn_syrup = 30 := by
  sorry

end water_amount_in_sport_formulation_l2417_241735


namespace area_traced_on_concentric_spheres_l2417_241734

/-- The area traced by a smaller sphere moving between two concentric spheres -/
theorem area_traced_on_concentric_spheres 
  (R1 R2 A1 : ℝ) 
  (h1 : 0 < R1) 
  (h2 : R1 < R2) 
  (h3 : 0 < A1) : 
  ∃ A2 : ℝ, A2 = A1 * (R2/R1)^2 := by
sorry

end area_traced_on_concentric_spheres_l2417_241734


namespace club_assignment_count_l2417_241701

/-- Represents the four clubs --/
inductive Club
| Literature
| Drama
| Anime
| Love

/-- Represents the five students --/
inductive Student
| A
| B
| C
| D
| E

/-- A valid club assignment is a function from Student to Club --/
def ClubAssignment := Student → Club

/-- Checks if a club assignment is valid according to the problem conditions --/
def is_valid_assignment (assignment : ClubAssignment) : Prop :=
  (∀ c : Club, ∃ s : Student, assignment s = c) ∧ 
  (assignment Student.A ≠ Club.Anime)

/-- The number of valid club assignments --/
def num_valid_assignments : ℕ := sorry

theorem club_assignment_count : num_valid_assignments = 180 := by sorry

end club_assignment_count_l2417_241701


namespace percentage_of_B_grades_l2417_241702

def scores : List ℕ := [88, 73, 55, 95, 76, 91, 86, 73, 76, 64, 85, 79, 72, 81, 89, 77]

def is_B_grade (score : ℕ) : Bool :=
  87 ≤ score ∧ score ≤ 94

def count_B_grades (scores : List ℕ) : ℕ :=
  scores.filter is_B_grade |>.length

theorem percentage_of_B_grades :
  (count_B_grades scores : ℚ) / scores.length * 100 = 12.5 := by sorry

end percentage_of_B_grades_l2417_241702


namespace side_b_value_l2417_241762

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Add conditions to ensure it's a valid triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi

-- State the theorem
theorem side_b_value (a b c : ℝ) (A B C : ℝ) :
  triangle_ABC a b c A B C →
  c = Real.sqrt 3 →
  B = Real.pi / 4 →
  C = Real.pi / 3 →
  b = Real.sqrt 2 := by
  sorry


end side_b_value_l2417_241762


namespace scientific_notation_of_448000_l2417_241765

/-- Proves that 448,000 is equal to 4.48 × 10^5 in scientific notation -/
theorem scientific_notation_of_448000 : 
  ∃ (a : ℝ) (n : ℤ), 448000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.48 ∧ n = 5 := by
  sorry

end scientific_notation_of_448000_l2417_241765


namespace sum_of_digits_less_than_1000_is_13500_l2417_241756

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def sum_of_digits_less_than_1000 : ℕ :=
  (List.range 1000).map digit_sum |> List.sum

theorem sum_of_digits_less_than_1000_is_13500 :
  sum_of_digits_less_than_1000 = 13500 := by
  sorry

end sum_of_digits_less_than_1000_is_13500_l2417_241756


namespace tuesday_rainfall_correct_l2417_241760

/-- Represents the rainfall data for three days -/
structure RainfallData where
  total : Float
  monday : Float
  wednesday : Float

/-- Calculates the rainfall on Tuesday given the rainfall data for three days -/
def tuesdayRainfall (data : RainfallData) : Float :=
  data.total - (data.monday + data.wednesday)

/-- Theorem stating that the rainfall on Tuesday is correctly calculated -/
theorem tuesday_rainfall_correct (data : RainfallData) 
  (h1 : data.total = 0.6666666666666666)
  (h2 : data.monday = 0.16666666666666666)
  (h3 : data.wednesday = 0.08333333333333333) :
  tuesdayRainfall data = 0.41666666666666663 := by
  sorry

#eval tuesdayRainfall { 
  total := 0.6666666666666666, 
  monday := 0.16666666666666666, 
  wednesday := 0.08333333333333333 
}

end tuesday_rainfall_correct_l2417_241760


namespace largest_power_of_18_dividing_30_factorial_l2417_241748

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem largest_power_of_18_dividing_30_factorial :
  (∃ n : ℕ, 18^n ∣ factorial 30 ∧ ∀ m : ℕ, m > n → ¬(18^m ∣ factorial 30)) →
  (∃ n : ℕ, n = 7 ∧ 18^n ∣ factorial 30 ∧ ∀ m : ℕ, m > n → ¬(18^m ∣ factorial 30)) :=
sorry

end largest_power_of_18_dividing_30_factorial_l2417_241748


namespace aron_vacuum_time_l2417_241743

/-- Represents the cleaning schedule and total cleaning time for Aron. -/
structure CleaningSchedule where
  vacuum_frequency : Nat  -- Number of days Aron vacuums per week
  dust_time : Nat         -- Minutes Aron spends dusting per day
  dust_frequency : Nat    -- Number of days Aron dusts per week
  total_cleaning_time : Nat  -- Total minutes Aron spends cleaning per week

/-- Calculates the number of minutes Aron spends vacuuming each day. -/
def vacuum_time_per_day (schedule : CleaningSchedule) : Nat :=
  (schedule.total_cleaning_time - schedule.dust_time * schedule.dust_frequency) / schedule.vacuum_frequency

/-- Theorem stating that Aron spends 30 minutes vacuuming each day. -/
theorem aron_vacuum_time (schedule : CleaningSchedule) 
    (h1 : schedule.vacuum_frequency = 3)
    (h2 : schedule.dust_time = 20)
    (h3 : schedule.dust_frequency = 2)
    (h4 : schedule.total_cleaning_time = 130) :
    vacuum_time_per_day schedule = 30 := by
  sorry


end aron_vacuum_time_l2417_241743


namespace rectangular_solid_volume_l2417_241727

theorem rectangular_solid_volume 
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 20)
  (h_front : front_area = 15)
  (h_bottom : bottom_area = 12) :
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    a * c = bottom_area ∧ 
    a * b * c = 60 := by
  sorry

end rectangular_solid_volume_l2417_241727
