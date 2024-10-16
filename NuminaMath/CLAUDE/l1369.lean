import Mathlib

namespace NUMINAMATH_CALUDE_cucumber_price_is_four_l1369_136978

/-- Represents the price of cucumbers per kilo -/
def cucumber_price : ℝ := sorry

/-- Theorem: Given Peter's shopping details, the price of cucumbers per kilo is $4 -/
theorem cucumber_price_is_four :
  let initial_amount : ℝ := 500
  let potatoes_kilo : ℝ := 6
  let potatoes_price : ℝ := 2
  let tomatoes_kilo : ℝ := 9
  let tomatoes_price : ℝ := 3
  let cucumbers_kilo : ℝ := 5
  let bananas_kilo : ℝ := 3
  let bananas_price : ℝ := 5
  let remaining_amount : ℝ := 426
  initial_amount - 
    (potatoes_kilo * potatoes_price + 
     tomatoes_kilo * tomatoes_price + 
     cucumbers_kilo * cucumber_price + 
     bananas_kilo * bananas_price) = remaining_amount →
  cucumber_price = 4 := by sorry

end NUMINAMATH_CALUDE_cucumber_price_is_four_l1369_136978


namespace NUMINAMATH_CALUDE_exists_function_1995_double_l1369_136944

/-- The number of iterations in the problem -/
def iterations : ℕ := 1995

/-- Definition of function iteration -/
def iterate (f : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, n => n
  | k + 1, n => f (iterate f k n)

/-- Theorem stating the existence of a function satisfying the condition -/
theorem exists_function_1995_double :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, iterate f iterations n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_exists_function_1995_double_l1369_136944


namespace NUMINAMATH_CALUDE_occur_permutations_correct_l1369_136933

/-- The number of unique permutations of the letters in "OCCUR" -/
def occurrPermutations : ℕ := 60

/-- The total number of letters in "OCCUR" -/
def totalLetters : ℕ := 5

/-- The number of times the letter "C" appears in "OCCUR" -/
def cCount : ℕ := 2

/-- Theorem stating that the number of unique permutations of "OCCUR" is correct -/
theorem occur_permutations_correct :
  occurrPermutations = (Nat.factorial totalLetters) / (Nat.factorial cCount) :=
by sorry

end NUMINAMATH_CALUDE_occur_permutations_correct_l1369_136933


namespace NUMINAMATH_CALUDE_binary_101_equals_5_l1369_136936

/-- Definition of binary to decimal conversion for a 3-digit binary number -/
def binary_to_decimal (b₂ : ℕ) (b₁ : ℕ) (b₀ : ℕ) : ℕ :=
  b₀ + 2 * b₁ + 4 * b₂

/-- Theorem stating that the binary number 101 is equal to the decimal number 5 -/
theorem binary_101_equals_5 : binary_to_decimal 1 0 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_equals_5_l1369_136936


namespace NUMINAMATH_CALUDE_locus_is_circle_l1369_136924

def locus_of_z (z₀ : ℂ) (z : ℂ) : Prop :=
  z₀ ≠ 0 ∧ z ≠ 0 ∧ ∃ z₁ : ℂ, Complex.abs (z₁ - z₀) = Complex.abs z₁ ∧ z₁ * z = -1

theorem locus_is_circle (z₀ : ℂ) (z : ℂ) :
  locus_of_z z₀ z → Complex.abs (z + 1 / z₀) = 1 / Complex.abs z₀ :=
by sorry

end NUMINAMATH_CALUDE_locus_is_circle_l1369_136924


namespace NUMINAMATH_CALUDE_pirate_treasure_year_l1369_136940

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base-8 representation of the year --/
def year_base8 : List Nat := [7, 6, 3]

/-- The claimed base-10 equivalent of the year --/
def year_base10 : Nat := 247

/-- Theorem stating that the base-8 year converts to the claimed base-10 year --/
theorem pirate_treasure_year : base8_to_base10 year_base8 = year_base10 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_year_l1369_136940


namespace NUMINAMATH_CALUDE_distribute_four_men_five_women_l1369_136994

/-- The number of ways to distribute men and women into groups -/
def distribute_people (num_men num_women : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem stating the correct number of distributions for 4 men and 5 women -/
theorem distribute_four_men_five_women :
  distribute_people 4 5 = 560 := by
  sorry

end NUMINAMATH_CALUDE_distribute_four_men_five_women_l1369_136994


namespace NUMINAMATH_CALUDE_inequality_proof_l1369_136957

theorem inequality_proof (b : ℝ) (n : ℕ) (h1 : b > 0) (h2 : n > 2) :
  let floor_b := ⌊b⌋
  let d := ((floor_b + 1 - b) * floor_b) / (floor_b + 1)
  (d + n - 2) / (floor_b + n - 2) > (floor_b + n - 1 - b) / (floor_b + n - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1369_136957


namespace NUMINAMATH_CALUDE_sum_N_equals_2250_l1369_136966

def N : ℕ → ℤ
  | 0 => 0
  | n + 1 => let k := 40 - n
              if n % 2 = 0 then
                N n + (3*k)^2 + (3*k-1)^2 + (3*k-2)^2
              else
                N n - (3*k)^2 - (3*k-1)^2 - (3*k-2)^2

theorem sum_N_equals_2250 : N 40 = 2250 := by
  sorry

end NUMINAMATH_CALUDE_sum_N_equals_2250_l1369_136966


namespace NUMINAMATH_CALUDE_car_rental_cost_l1369_136973

/-- Calculates the total cost of renting a car for three days with varying rates and mileage. -/
theorem car_rental_cost (base_rate_day1 base_rate_day2 base_rate_day3 : ℚ)
                        (per_mile_day1 per_mile_day2 per_mile_day3 : ℚ)
                        (miles_day1 miles_day2 miles_day3 : ℚ) :
  base_rate_day1 = 150 →
  base_rate_day2 = 100 →
  base_rate_day3 = 75 →
  per_mile_day1 = 0.5 →
  per_mile_day2 = 0.4 →
  per_mile_day3 = 0.3 →
  miles_day1 = 620 →
  miles_day2 = 744 →
  miles_day3 = 510 →
  base_rate_day1 + miles_day1 * per_mile_day1 +
  base_rate_day2 + miles_day2 * per_mile_day2 +
  base_rate_day3 + miles_day3 * per_mile_day3 = 1085.6 :=
by sorry

end NUMINAMATH_CALUDE_car_rental_cost_l1369_136973


namespace NUMINAMATH_CALUDE_school_trip_ratio_l1369_136951

theorem school_trip_ratio (total : ℕ) (remaining : ℕ) : 
  total = 1000 → 
  remaining = 250 → 
  (total / 2 - remaining) / remaining = 1 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_ratio_l1369_136951


namespace NUMINAMATH_CALUDE_smallest_n_square_and_fifth_power_l1369_136901

theorem smallest_n_square_and_fifth_power :
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (l : ℕ), 5 * n = l^5) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (k : ℕ), 4 * m = k^2) → 
    (∃ (l : ℕ), 5 * m = l^5) → 
    m ≥ 625) ∧
  n = 625 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_fifth_power_l1369_136901


namespace NUMINAMATH_CALUDE_probability_one_or_two_pascal_l1369_136931

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) := sorry

/-- Counts the occurrences of a specific value in Pascal's Triangle up to n rows -/
def countOccurrences (n : ℕ) (value : ℕ) : ℕ := sorry

/-- Calculates the total number of elements in Pascal's Triangle up to n rows -/
def totalElements (n : ℕ) : ℕ := sorry

/-- The main theorem stating the probability of selecting 1 or 2 from the first 20 rows of Pascal's Triangle -/
theorem probability_one_or_two_pascal : 
  (countOccurrences 20 1 + countOccurrences 20 2) / totalElements 20 = 37 / 105 := by sorry

end NUMINAMATH_CALUDE_probability_one_or_two_pascal_l1369_136931


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1369_136998

theorem perfect_square_trinomial (a b c : ℤ) :
  (∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^2) →
  ∃ d e : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (d * x + e)^2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1369_136998


namespace NUMINAMATH_CALUDE_polynomial_root_l1369_136916

/-- Given a polynomial g(x) = 3x^4 - 2x^3 + x^2 + 4x + s, 
    prove that s = -2 when g(-1) = 0 -/
theorem polynomial_root (s : ℝ) : 
  (fun x : ℝ => 3 * x^4 - 2 * x^3 + x^2 + 4 * x + s) (-1) = 0 ↔ s = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_l1369_136916


namespace NUMINAMATH_CALUDE_system_solution_unique_l1369_136905

theorem system_solution_unique (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 / x + 2 / y = 4 ∧ 5 / x - 6 / y = 2) ↔ (x = 1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1369_136905


namespace NUMINAMATH_CALUDE_unicorns_games_played_l1369_136972

theorem unicorns_games_played (initial_games : ℕ) (initial_wins : ℕ) : 
  initial_wins = (initial_games * 45 / 100) →
  (initial_wins + 6) = ((initial_games + 8) * 1 / 2) →
  initial_games + 8 = 48 := by
sorry

end NUMINAMATH_CALUDE_unicorns_games_played_l1369_136972


namespace NUMINAMATH_CALUDE_sqrt_of_negative_nine_squared_l1369_136903

theorem sqrt_of_negative_nine_squared : Real.sqrt ((-9)^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_negative_nine_squared_l1369_136903


namespace NUMINAMATH_CALUDE_antonette_age_l1369_136945

theorem antonette_age (a t : ℝ) 
  (h1 : t = 3 * a)  -- Tom is thrice as old as Antonette
  (h2 : a + t = 54) -- The sum of their ages is 54
  : a = 13.5 := by  -- Prove that Antonette's age is 13.5
  sorry

end NUMINAMATH_CALUDE_antonette_age_l1369_136945


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1369_136908

theorem inequality_solution_set (x : ℝ) :
  (Set.Ioo (-4 : ℝ) 2) = {x | (2 - x) / (x + 4) > 0} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1369_136908


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1369_136925

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → (a 3 + a 4 + a 5 = 12) → (a 1 + a 7 = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1369_136925


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l1369_136900

/-- Given an arithmetic sequence with first term 2 and common difference 3,
    prove that the 8th term is 23. -/
theorem arithmetic_sequence_8th_term :
  let a : ℕ → ℤ := λ n => 2 + 3 * (n - 1)
  a 8 = 23 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l1369_136900


namespace NUMINAMATH_CALUDE_solve_equation_l1369_136983

theorem solve_equation (x : ℝ) (h : (0.12 / x) * 2 = 12) : x = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1369_136983


namespace NUMINAMATH_CALUDE_initial_amount_correct_l1369_136956

/-- The amount of money John initially gave when buying barbells -/
def initial_amount : ℕ := 850

/-- The number of barbells John bought -/
def num_barbells : ℕ := 3

/-- The cost of each barbell in dollars -/
def barbell_cost : ℕ := 270

/-- The amount of change John received in dollars -/
def change_received : ℕ := 40

/-- Theorem stating that the initial amount John gave is correct -/
theorem initial_amount_correct : 
  initial_amount = num_barbells * barbell_cost + change_received :=
by sorry

end NUMINAMATH_CALUDE_initial_amount_correct_l1369_136956


namespace NUMINAMATH_CALUDE_smaller_hexagon_area_ratio_l1369_136950

/-- A regular hexagon with side length 4 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 4)

/-- Midpoint of a side of the hexagon -/
structure Midpoint :=
  (point : ℝ × ℝ)

/-- The smaller hexagon formed by connecting midpoints of alternating sides -/
structure SmallerHexagon :=
  (vertices : List (ℝ × ℝ))
  (is_regular : Bool)

/-- The ratio of the area of the smaller hexagon to the area of the original hexagon -/
def area_ratio (original : RegularHexagon) (smaller : SmallerHexagon) : ℚ :=
  49/36

theorem smaller_hexagon_area_ratio 
  (original : RegularHexagon) 
  (G H I J K L : Midpoint) 
  (smaller : SmallerHexagon) :
  area_ratio original smaller = 49/36 :=
sorry

end NUMINAMATH_CALUDE_smaller_hexagon_area_ratio_l1369_136950


namespace NUMINAMATH_CALUDE_probability_of_drawing_red_ball_l1369_136928

theorem probability_of_drawing_red_ball (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) :
  total_balls = red_balls + black_balls →
  red_balls = 3 →
  black_balls = 3 →
  (red_balls : ℚ) / (total_balls : ℚ) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_drawing_red_ball_l1369_136928


namespace NUMINAMATH_CALUDE_unique_perpendicular_tangent_perpendicular_tangent_equation_angle_of_inclination_range_l1369_136911

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

-- Statement for the unique perpendicular tangent line
theorem unique_perpendicular_tangent (a : ℝ) :
  (∃! x : ℝ, f' a x = -1) ↔ a = 3 :=
sorry

-- Statement for the equation of the perpendicular tangent line
theorem perpendicular_tangent_equation (a : ℝ) (h : a = 3) :
  ∃ x y : ℝ, 3*x + y - 8 = 0 ∧ y = f a x ∧ f' a x = -1 :=
sorry

-- Statement for the range of the angle of inclination
theorem angle_of_inclination_range (a : ℝ) (h : a = 3) :
  ∀ x : ℝ, -π/4 < Real.arctan (f' a x) ∧ Real.arctan (f' a x) < π/2 :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_tangent_perpendicular_tangent_equation_angle_of_inclination_range_l1369_136911


namespace NUMINAMATH_CALUDE_polynomial_property_l1369_136913

-- Define the polynomial Q(x)
def Q (x d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

-- Define the conditions
theorem polynomial_property (d e f : ℝ) :
  -- The y-intercept is 12
  Q 0 d e f = 12 →
  -- The product of zeros is equal to -f/3
  (∃ α β γ : ℝ, Q α d e f = 0 ∧ Q β d e f = 0 ∧ Q γ d e f = 0 ∧ α * β * γ = -f / 3) →
  -- The mean of zeros is equal to the product of zeros
  (∃ α β γ : ℝ, Q α d e f = 0 ∧ Q β d e f = 0 ∧ Q γ d e f = 0 ∧ (α + β + γ) / 3 = -f / 3) →
  -- The sum of coefficients is equal to the product of zeros
  3 + d + e + f = -f / 3 →
  -- Then e = -55
  e = -55 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_property_l1369_136913


namespace NUMINAMATH_CALUDE_cut_cylinder_unpainted_face_area_l1369_136981

/-- The area of an unpainted face of a cut cylinder -/
theorem cut_cylinder_unpainted_face_area (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  let sector_area := π * r^2 / 4
  let triangle_area := r^2 / 2
  let unpainted_face_area := h * (sector_area + triangle_area)
  unpainted_face_area = 62.5 * π + 125 := by
  sorry

end NUMINAMATH_CALUDE_cut_cylinder_unpainted_face_area_l1369_136981


namespace NUMINAMATH_CALUDE_ones_12_div_13_ones_16_div_17_l1369_136927

/-- The number formed by n consecutive ones -/
def ones (n : ℕ) : ℕ := (10^n - 1) / 9

/-- Theorem: The number formed by 12 consecutive ones is divisible by 13 -/
theorem ones_12_div_13 : 13 ∣ ones 12 := by sorry

/-- Theorem: The number formed by 16 consecutive ones is divisible by 17 -/
theorem ones_16_div_17 : 17 ∣ ones 16 := by sorry

end NUMINAMATH_CALUDE_ones_12_div_13_ones_16_div_17_l1369_136927


namespace NUMINAMATH_CALUDE_joe_weight_lifting_problem_l1369_136939

theorem joe_weight_lifting_problem (total_weight first_lift_weight : ℕ) 
  (h1 : total_weight = 900)
  (h2 : first_lift_weight = 400) : 
  2 * first_lift_weight - (total_weight - first_lift_weight) = 300 := by
  sorry

end NUMINAMATH_CALUDE_joe_weight_lifting_problem_l1369_136939


namespace NUMINAMATH_CALUDE_red_spot_percentage_is_40_l1369_136902

/-- Represents the farm with cows and their spot characteristics -/
structure Farm where
  total_cows : ℕ
  no_spot_cows : ℕ
  blue_spot_ratio : ℚ

/-- Calculates the percentage of cows with a red spot -/
def red_spot_percentage (farm : Farm) : ℚ :=
  let no_red_spot := farm.no_spot_cows / farm.blue_spot_ratio
  let red_spot := farm.total_cows - no_red_spot
  (red_spot / farm.total_cows) * 100

/-- Theorem stating that for the given farm conditions, 
    the percentage of cows with a red spot is 40% -/
theorem red_spot_percentage_is_40 (farm : Farm) 
  (h1 : farm.total_cows = 140)
  (h2 : farm.no_spot_cows = 63)
  (h3 : farm.blue_spot_ratio = 3/4) :
  red_spot_percentage farm = 40 := by
  sorry

#eval red_spot_percentage ⟨140, 63, 3/4⟩

end NUMINAMATH_CALUDE_red_spot_percentage_is_40_l1369_136902


namespace NUMINAMATH_CALUDE_decimal_sum_difference_l1369_136941

theorem decimal_sum_difference : 0.5 - 0.03 + 0.007 + 0.0008 = 0.4778 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_difference_l1369_136941


namespace NUMINAMATH_CALUDE_ribbon_length_problem_l1369_136915

theorem ribbon_length_problem (a b : ℕ+) (ha : a = 8) (h_gcd : Nat.gcd a b = 8) : b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_problem_l1369_136915


namespace NUMINAMATH_CALUDE_four_inch_cube_painted_faces_l1369_136976

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Represents a painted cube -/
structure PaintedCube extends Cube where
  paintedFaces : ℕ

/-- Calculates the number of smaller cubes with at least two painted faces
    when a painted cube is cut into unit cubes -/
def numCubesWithTwoPaintedFaces (c : PaintedCube) : ℕ :=
  sorry

theorem four_inch_cube_painted_faces :
  let bigCube : PaintedCube := ⟨⟨4⟩, 6⟩
  numCubesWithTwoPaintedFaces bigCube = 32 := by sorry

end NUMINAMATH_CALUDE_four_inch_cube_painted_faces_l1369_136976


namespace NUMINAMATH_CALUDE_projection_vector_l1369_136999

/-- Two parallel lines r and s in 2D space -/
structure ParallelLines where
  r : ℝ → ℝ × ℝ
  s : ℝ → ℝ × ℝ
  hr : ∀ t, r t = (2 + 5*t, 3 - 2*t)
  hs : ∀ u, s u = (1 + 5*u, -2 - 2*u)

/-- Points C, D, and Q in 2D space -/
structure Points (l : ParallelLines) where
  C : ℝ × ℝ
  D : ℝ × ℝ
  Q : ℝ × ℝ
  hC : ∃ t, l.r t = C
  hD : ∃ u, l.s u = D
  hQ : (Q.1 - C.1) * 5 + (Q.2 - C.2) * (-2) = 0 -- Q is on the perpendicular to s passing through C

/-- The theorem to be proved -/
theorem projection_vector (l : ParallelLines) (p : Points l) :
  ∃ k : ℝ, 
    (p.Q.1 - p.C.1, p.Q.2 - p.C.2) = k • (-2, -5) ∧
    (p.D.1 - p.C.1) * (-2) + (p.D.2 - p.C.2) * (-5) = 
      (p.Q.1 - p.C.1) * (-2) + (p.Q.2 - p.C.2) * (-5) ∧
    -2 - (-5) = 3 :=
  sorry

end NUMINAMATH_CALUDE_projection_vector_l1369_136999


namespace NUMINAMATH_CALUDE_sum_of_sequence_l1369_136958

/-- Calculates the sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

/-- The number of terms in the sequence -/
def num_terms (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

theorem sum_of_sequence : 
  let a₁ := 71  -- First term
  let aₙ := 361 -- Last term
  let d := 10   -- Common difference
  let n := num_terms a₁ aₙ d
  arithmetic_sum a₁ aₙ n = 6480 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_sequence_l1369_136958


namespace NUMINAMATH_CALUDE_total_books_l1369_136953

theorem total_books (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : books_per_shelf = 6)
  (h2 : mystery_shelves = 5)
  (h3 : picture_shelves = 4) :
  books_per_shelf * mystery_shelves + books_per_shelf * picture_shelves = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l1369_136953


namespace NUMINAMATH_CALUDE_first_day_over_500_l1369_136963

def paperclips (k : ℕ) : ℕ := 4 * 3^k

theorem first_day_over_500 : 
  (∃ k : ℕ, paperclips k > 500) ∧ 
  (∀ j : ℕ, j < 5 → paperclips j ≤ 500) ∧ 
  (paperclips 5 > 500) :=
by sorry

end NUMINAMATH_CALUDE_first_day_over_500_l1369_136963


namespace NUMINAMATH_CALUDE_cubic_root_odd_and_increasing_l1369_136986

-- Define the function
def f (x : ℝ) : ℝ := x^(1/3)

-- State the theorem
theorem cubic_root_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_odd_and_increasing_l1369_136986


namespace NUMINAMATH_CALUDE_inverse_scalar_multiple_l1369_136960

/-- Given a 2x2 matrix B and a constant l, prove that B^(-1) = l * B implies e = -3 and l = 1/19 -/
theorem inverse_scalar_multiple (B : Matrix (Fin 2) (Fin 2) ℝ) (l : ℝ) :
  B 0 0 = 3 ∧ B 0 1 = 4 ∧ B 1 0 = 7 ∧ B 1 1 = B.det / (3 * B 1 1 - 28) →
  B⁻¹ = l • B →
  B 1 1 = -3 ∧ l = 1 / 19 := by
sorry

end NUMINAMATH_CALUDE_inverse_scalar_multiple_l1369_136960


namespace NUMINAMATH_CALUDE_cube_cannot_cover_5x5_square_l1369_136946

/-- Represents the four possible directions on a chessboard -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents the state of the cube -/
structure CubeState :=
  (position : Position)
  (topFace : Fin 6)
  (faceDirections : Fin 6 → Direction)

/-- The set of all positions a cube can visit given its initial state -/
def visitablePositions (initialState : CubeState) : Set Position :=
  sorry

/-- A 5x5 square on the chessboard -/
def square5x5 (topLeft : Position) : Set Position :=
  { p : Position | 
    topLeft.x ≤ p.x ∧ p.x < topLeft.x + 5 ∧
    topLeft.y - 4 ≤ p.y ∧ p.y ≤ topLeft.y }

/-- Theorem stating that the cube cannot cover any 5x5 square -/
theorem cube_cannot_cover_5x5_square (initialState : CubeState) :
  ∀ topLeft : Position, ¬(square5x5 topLeft ⊆ visitablePositions initialState) :=
sorry

end NUMINAMATH_CALUDE_cube_cannot_cover_5x5_square_l1369_136946


namespace NUMINAMATH_CALUDE_root_existence_condition_l1369_136920

def f (m : ℝ) (x : ℝ) : ℝ := m * x + 6

theorem root_existence_condition (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 3, f m x = 0) ↔ m ≤ -2 ∨ m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_root_existence_condition_l1369_136920


namespace NUMINAMATH_CALUDE_cylinder_generatrix_length_l1369_136975

/-- The length of the generatrix of a cylinder with base radius 1 and lateral surface area 6π is 2 -/
theorem cylinder_generatrix_length :
  ∀ (generatrix : ℝ),
  (generatrix > 0) →
  (2 * π * 1 + 2 * π * generatrix = 6 * π) →
  generatrix = 2 := by
sorry

end NUMINAMATH_CALUDE_cylinder_generatrix_length_l1369_136975


namespace NUMINAMATH_CALUDE_min_tangent_length_is_4_l1369_136982

/-- The circle C with equation x^2 + y^2 + 2x - 4y + 3 = 0 -/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 3 = 0

/-- The line of symmetry for circle C -/
def symmetry_line (a b x y : ℝ) : Prop :=
  2*a*x + b*y + 6 = 0

/-- The point (a, b) -/
structure Point where
  a : ℝ
  b : ℝ

/-- The minimum tangent length from a point to a circle -/
def min_tangent_length (p : Point) (C : (ℝ → ℝ → Prop)) : ℝ :=
  sorry

theorem min_tangent_length_is_4 (a b : ℝ) :
  symmetry_line a b a b →
  min_tangent_length (Point.mk a b) circle_C = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_tangent_length_is_4_l1369_136982


namespace NUMINAMATH_CALUDE_college_student_count_l1369_136979

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- The total number of students in the college -/
def College.total (c : College) : ℕ := c.boys + c.girls

/-- Theorem: In a college where the ratio of boys to girls is 8:5 and there are 300 girls, 
    the total number of students is 780 -/
theorem college_student_count : 
  ∀ (c : College), 
  c.boys * 5 = c.girls * 8 → 
  c.girls = 300 → 
  c.total = 780 := by
sorry

end NUMINAMATH_CALUDE_college_student_count_l1369_136979


namespace NUMINAMATH_CALUDE_floor_neg_three_point_seven_l1369_136970

-- Define the greatest integer function
def floor (x : ℝ) : ℤ := sorry

-- State the theorem
theorem floor_neg_three_point_seven :
  floor (-3.7) = -4 := by sorry

end NUMINAMATH_CALUDE_floor_neg_three_point_seven_l1369_136970


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l1369_136943

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

def monotonic_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

theorem monotonic_decreasing_interval :
  ∀ x, x > 0 → (monotonic_decreasing f 0 1) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l1369_136943


namespace NUMINAMATH_CALUDE_quadratic_points_property_l1369_136917

/-- Represents a quadratic function y = ax² - 4ax + c, where a > 0 -/
structure QuadraticFunction where
  a : ℝ
  c : ℝ
  h_a_pos : a > 0

/-- Represents the y-coordinates of the four points on the quadratic function -/
structure FourPoints where
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  y₄ : ℝ

/-- 
  Theorem: For a quadratic function y = ax² - 4ax + c (a > 0) passing through points 
  A(-2, y₁), B(0, y₂), C(3, y₃), D(5, y₄), if y₂y₄ < 0, then y₁y₃ < 0.
-/
theorem quadratic_points_property (f : QuadraticFunction) (p : FourPoints) :
  (p.y₂ * p.y₄ < 0) → (p.y₁ * p.y₃ < 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_points_property_l1369_136917


namespace NUMINAMATH_CALUDE_promotion_payment_correct_l1369_136984

/-- Represents the payment calculation for a clothing factory promotion -/
def promotion_payment (suit_price tie_price : ℕ) (num_suits num_ties : ℕ) : ℕ × ℕ :=
  let option1 := suit_price * num_suits + tie_price * (num_ties - num_suits)
  let option2 := ((suit_price * num_suits + tie_price * num_ties) * 9) / 10
  (option1, option2)

theorem promotion_payment_correct (x : ℕ) (h : x > 20) :
  promotion_payment 200 40 20 x = (40 * x + 3200, 3600 + 36 * x) := by
  sorry

#eval promotion_payment 200 40 20 30

end NUMINAMATH_CALUDE_promotion_payment_correct_l1369_136984


namespace NUMINAMATH_CALUDE_ludvik_favorite_number_l1369_136929

/-- Ludvík's favorite number problem -/
theorem ludvik_favorite_number 
  (a b : ℝ) -- original dividend and divisor
  (h1 : (2 * a) / (b + 12) = (a - 42) / (b / 2)) -- equality of the two scenarios
  (h2 : (2 * a) / (b + 12) > 0) -- ensure the result is positive
  : (2 * a) / (b + 12) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ludvik_favorite_number_l1369_136929


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l1369_136904

theorem point_movement_on_number_line (A : ℝ) (movement : ℝ) : 
  A = -2 → movement = 4 → (A - movement = -6 ∨ A + movement = 2) := by
  sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l1369_136904


namespace NUMINAMATH_CALUDE_divisibility_condition_l1369_136948

theorem divisibility_condition (n : ℕ) : (n + 1) ∣ (n^2 + 1) ↔ n = 0 ∨ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1369_136948


namespace NUMINAMATH_CALUDE_perfect_square_characterization_l1369_136937

theorem perfect_square_characterization (A : ℕ) :
  (∃ k : ℕ, A = k^2) ↔
  (∀ n : ℕ, ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ n ∣ ((A + i)^2 - A)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_characterization_l1369_136937


namespace NUMINAMATH_CALUDE_product_equals_120_l1369_136949

theorem product_equals_120 (n : ℕ) (h : n = 3) : (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_120_l1369_136949


namespace NUMINAMATH_CALUDE_magazine_cost_l1369_136967

theorem magazine_cost (total_books : ℕ) (num_magazines : ℕ) (book_cost : ℕ) (total_spent : ℕ) : 
  total_books = 16 → 
  num_magazines = 3 → 
  book_cost = 11 → 
  total_spent = 179 → 
  (total_spent - total_books * book_cost) / num_magazines = 1 := by
sorry

end NUMINAMATH_CALUDE_magazine_cost_l1369_136967


namespace NUMINAMATH_CALUDE_cool_drink_jasmine_percentage_l1369_136910

/-- Represents the initial percentage of jasmine water in the solution -/
def initial_percentage : ℝ := 5

/-- The initial volume of the solution in liters -/
def initial_volume : ℝ := 90

/-- The volume of jasmine added in liters -/
def added_jasmine : ℝ := 8

/-- The volume of water added in liters -/
def added_water : ℝ := 2

/-- The final percentage of jasmine in the solution -/
def final_percentage : ℝ := 12.5

/-- The final volume of the solution in liters -/
def final_volume : ℝ := initial_volume + added_jasmine + added_water

theorem cool_drink_jasmine_percentage :
  (initial_percentage / 100) * initial_volume + added_jasmine = 
  (final_percentage / 100) * final_volume :=
sorry

end NUMINAMATH_CALUDE_cool_drink_jasmine_percentage_l1369_136910


namespace NUMINAMATH_CALUDE_flower_expenses_l1369_136938

/-- The total expenses for ordering flowers at Parc Municipal -/
theorem flower_expenses : 
  let tulips : ℕ := 250
  let carnations : ℕ := 375
  let roses : ℕ := 320
  let price_per_flower : ℕ := 2
  (tulips + carnations + roses) * price_per_flower = 1890 := by
  sorry

end NUMINAMATH_CALUDE_flower_expenses_l1369_136938


namespace NUMINAMATH_CALUDE_not_arithmetic_sequence_l1369_136909

theorem not_arithmetic_sequence : ¬∃ (a d : ℝ) (m n k : ℤ), 
  a + (m - 1 : ℝ) * d = 1 ∧ 
  a + (n - 1 : ℝ) * d = Real.sqrt 2 ∧ 
  a + (k - 1 : ℝ) * d = 3 ∧ 
  n = m + 1 ∧ 
  k = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_not_arithmetic_sequence_l1369_136909


namespace NUMINAMATH_CALUDE_joao_chocolate_bars_l1369_136989

theorem joao_chocolate_bars (x y z : ℕ) : 
  x + y + z = 30 →
  2 * x + 3 * y + 4 * z = 100 →
  z > x :=
by sorry

end NUMINAMATH_CALUDE_joao_chocolate_bars_l1369_136989


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1369_136919

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 + Real.sqrt (2 * x - 1)) = 4 → x = 85 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1369_136919


namespace NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_1_l1369_136962

theorem factorization_of_4x_squared_minus_1 (x : ℝ) : 4 * x^2 - 1 = (2*x + 1) * (2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_1_l1369_136962


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l1369_136930

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l1369_136930


namespace NUMINAMATH_CALUDE_hardware_store_earnings_l1369_136921

/-- Calculates the total earnings of a hardware store for a week given the sales and prices of various items. -/
theorem hardware_store_earnings 
  (graphics_cards_sold : ℕ) (graphics_card_price : ℕ)
  (hard_drives_sold : ℕ) (hard_drive_price : ℕ)
  (cpus_sold : ℕ) (cpu_price : ℕ)
  (ram_pairs_sold : ℕ) (ram_pair_price : ℕ)
  (h1 : graphics_cards_sold = 10)
  (h2 : graphics_card_price = 600)
  (h3 : hard_drives_sold = 14)
  (h4 : hard_drive_price = 80)
  (h5 : cpus_sold = 8)
  (h6 : cpu_price = 200)
  (h7 : ram_pairs_sold = 4)
  (h8 : ram_pair_price = 60) :
  graphics_cards_sold * graphics_card_price +
  hard_drives_sold * hard_drive_price +
  cpus_sold * cpu_price +
  ram_pairs_sold * ram_pair_price = 8960 := by
  sorry

end NUMINAMATH_CALUDE_hardware_store_earnings_l1369_136921


namespace NUMINAMATH_CALUDE_tangent_line_x_ln_x_l1369_136980

/-- The equation of the tangent line to y = x ln x at (1, 0) is x - y - 1 = 0 -/
theorem tangent_line_x_ln_x (x y : ℝ) : 
  (∀ t, t > 0 → y = t * Real.log t) →  -- Definition of the curve
  (x = 1 ∧ y = 0) →                    -- Point of tangency
  (x - y - 1 = 0) :=                   -- Equation of the tangent line
by sorry

end NUMINAMATH_CALUDE_tangent_line_x_ln_x_l1369_136980


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_63_2898_l1369_136934

theorem gcd_lcm_sum_63_2898 : Nat.gcd 63 2898 + Nat.lcm 63 2898 = 182575 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_63_2898_l1369_136934


namespace NUMINAMATH_CALUDE_remaining_savings_is_25_70_l1369_136997

/-- Calculates the remaining savings after jewelry purchases and tax --/
def remaining_savings (initial_savings earrings_cost necklace_cost bracelet_cost jewelry_set_cost discount_percent tax_percent : ℚ) : ℚ :=
  let individual_items_cost := earrings_cost + necklace_cost + bracelet_cost
  let discounted_jewelry_set_cost := jewelry_set_cost * (1 - discount_percent / 100)
  let total_before_tax := individual_items_cost + discounted_jewelry_set_cost
  let tax_amount := total_before_tax * (tax_percent / 100)
  let final_total_cost := total_before_tax + tax_amount
  initial_savings - final_total_cost

/-- Theorem stating that the remaining savings are $25.70 --/
theorem remaining_savings_is_25_70 :
  remaining_savings 200 23 48 35 80 25 5 = 25.70 := by
  sorry

end NUMINAMATH_CALUDE_remaining_savings_is_25_70_l1369_136997


namespace NUMINAMATH_CALUDE_nail_count_l1369_136968

/-- Given that Violet has 3 more than twice as many nails as Tickletoe and Violet has 27 nails, 
    prove that the total number of nails they have together is 39. -/
theorem nail_count (tickletoe_nails : ℕ) : 
  (2 * tickletoe_nails + 3 = 27) → (tickletoe_nails + 27 = 39) := by
  sorry

end NUMINAMATH_CALUDE_nail_count_l1369_136968


namespace NUMINAMATH_CALUDE_xy_equals_five_l1369_136995

theorem xy_equals_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hdistinct : x ≠ y)
  (h : x + 5 / x = y + 5 / y) : x * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_five_l1369_136995


namespace NUMINAMATH_CALUDE_simplify_power_of_product_l1369_136961

theorem simplify_power_of_product (y : ℝ) : (3 * y^4)^2 = 9 * y^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_of_product_l1369_136961


namespace NUMINAMATH_CALUDE_weight_sum_l1369_136955

/-- Given the weights of four people satisfying certain conditions, 
    prove that the sum of the first and fourth person's weights is 372 pounds. -/
theorem weight_sum (e f g h : ℝ) 
  (ef_sum : e + f = 320)
  (fg_sum : f + g = 298)
  (gh_sum : g + h = 350) :
  e + h = 372 := by
  sorry

end NUMINAMATH_CALUDE_weight_sum_l1369_136955


namespace NUMINAMATH_CALUDE_min_rain_day4_exceeds_21_inches_l1369_136971

/-- Represents the rainfall and drainage scenario over 4 days -/
structure RainfallScenario where
  capacity : ℝ  -- Total capacity in inches
  drainRate : ℝ  -- Drainage rate in inches per day
  day1Rain : ℝ  -- Rainfall on day 1 in inches
  day2Rain : ℝ  -- Rainfall on day 2 in inches
  day3Rain : ℝ  -- Rainfall on day 3 in inches

/-- Calculates the minimum rainfall on day 4 to cause flooding -/
def minRainDay4ToFlood (scenario : RainfallScenario) : ℝ :=
  scenario.capacity - (scenario.day1Rain + scenario.day2Rain + scenario.day3Rain - 3 * scenario.drainRate)

/-- Theorem stating the minimum rainfall on day 4 to cause flooding is more than 21 inches -/
theorem min_rain_day4_exceeds_21_inches (scenario : RainfallScenario) 
    (h1 : scenario.capacity = 72) -- 6 feet = 72 inches
    (h2 : scenario.drainRate = 3)
    (h3 : scenario.day1Rain = 10)
    (h4 : scenario.day2Rain = 2 * scenario.day1Rain)
    (h5 : scenario.day3Rain = 1.5 * scenario.day2Rain) : 
  minRainDay4ToFlood scenario > 21 := by
  sorry

#eval minRainDay4ToFlood { capacity := 72, drainRate := 3, day1Rain := 10, day2Rain := 20, day3Rain := 30 }

end NUMINAMATH_CALUDE_min_rain_day4_exceeds_21_inches_l1369_136971


namespace NUMINAMATH_CALUDE_transaction_result_l1369_136987

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  carValue : Int
  hascar : Bool

/-- Represents a car transaction between two people -/
def carTransaction (buyer seller : FinancialState) (price : Int) : FinancialState × FinancialState :=
  let newBuyer : FinancialState := {
    cash := buyer.cash - price,
    carValue := seller.carValue,
    hascar := true
  }
  let newSeller : FinancialState := {
    cash := seller.cash + price,
    carValue := 0,
    hascar := false
  }
  (newBuyer, newSeller)

/-- Calculates the net worth of a person -/
def netWorth (state : FinancialState) : Int :=
  state.cash + (if state.hascar then state.carValue else 0)

theorem transaction_result (initialCarValue : Int) :
  let mrAInitial : FinancialState := { cash := 8000, carValue := initialCarValue, hascar := true }
  let mrBInitial : FinancialState := { cash := 9000, carValue := 0, hascar := false }
  let (mrBAfterFirst, mrAAfterFirst) := carTransaction mrBInitial mrAInitial 10000
  let (mrAFinal, mrBFinal) := carTransaction mrAAfterFirst mrBAfterFirst 7000
  (netWorth mrAFinal - netWorth mrAInitial = 3000) ∧
  (netWorth mrBFinal - netWorth mrBInitial = -3000) :=
by
  sorry


end NUMINAMATH_CALUDE_transaction_result_l1369_136987


namespace NUMINAMATH_CALUDE_tori_trash_total_l1369_136996

/-- The number of pieces of trash Tori picked up in the classrooms -/
def classroom_trash : ℕ := 344

/-- The number of pieces of trash Tori picked up outside the classrooms -/
def outside_trash : ℕ := 1232

/-- The total number of pieces of trash Tori picked up last week -/
def total_trash : ℕ := classroom_trash + outside_trash

/-- Theorem stating that the total number of pieces of trash Tori picked up is 1576 -/
theorem tori_trash_total : total_trash = 1576 := by
  sorry

end NUMINAMATH_CALUDE_tori_trash_total_l1369_136996


namespace NUMINAMATH_CALUDE_exists_prime_with_greater_remainder_l1369_136947

theorem exists_prime_with_greater_remainder
  (a b : ℕ+) (h : a < b) :
  ∃ p : ℕ, Nat.Prime p ∧ a % p > b % p :=
sorry

end NUMINAMATH_CALUDE_exists_prime_with_greater_remainder_l1369_136947


namespace NUMINAMATH_CALUDE_product_equals_243_l1369_136942

theorem product_equals_243 : 
  (1/3 : ℚ) * 9 * (1/27 : ℚ) * 81 * (1/243 : ℚ) * 729 * (1/2187 : ℚ) * 6561 * (1/19683 : ℚ) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_243_l1369_136942


namespace NUMINAMATH_CALUDE_projection_is_regular_polygon_l1369_136974

-- Define the types of polyhedra
inductive Polyhedron
  | Dodecahedron
  | Icosahedron

-- Define a regular polygon
structure RegularPolygon where
  sides : ℕ
  is_regular : Bool

-- Define a projection function
def project (p : Polyhedron) : RegularPolygon :=
  match p with
  | Polyhedron.Dodecahedron => { sides := 10, is_regular := true }
  | Polyhedron.Icosahedron => { sides := 6, is_regular := true }

-- Theorem statement
theorem projection_is_regular_polygon (p : Polyhedron) :
  (project p).is_regular = true :=
by sorry

end NUMINAMATH_CALUDE_projection_is_regular_polygon_l1369_136974


namespace NUMINAMATH_CALUDE_complex_equation_proof_l1369_136969

/-- Given the complex equation (2+i)/(i+1) - 2i = a + bi, prove that b - ai = -5/2 - 3/2i --/
theorem complex_equation_proof (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (2 + i) / (i + 1) - 2 * i = a + b * i) : 
  b - a * i = -5/2 - 3/2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l1369_136969


namespace NUMINAMATH_CALUDE_yoongis_number_l1369_136977

theorem yoongis_number (x : ℤ) (h : x - 10 = 15) : x + 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_yoongis_number_l1369_136977


namespace NUMINAMATH_CALUDE_h_degree_three_iff_c_eq_three_l1369_136918

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 3 - 8*x + 2*x^2 - 7*x^3 + 6*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 2 - 3*x + x^3 - 2*x^4

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * (g x)

/-- Theorem stating that h(x) has degree 3 if and only if c = 3 -/
theorem h_degree_three_iff_c_eq_three :
  ∃! c : ℝ, (∀ x : ℝ, h c x = 3 - 8*x + 2*x^2 - 4*x^3) ∧ c = 3 :=
sorry

end NUMINAMATH_CALUDE_h_degree_three_iff_c_eq_three_l1369_136918


namespace NUMINAMATH_CALUDE_ratio_problem_l1369_136914

theorem ratio_problem (A B C : ℝ) (h1 : A + B + C = 98) (h2 : A / B = 2 / 3) (h3 : B = 30) :
  B / C = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1369_136914


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_square_lt_one_l1369_136923

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := by sorry

theorem negation_of_square_lt_one :
  (¬ ∃ x : ℝ, x^2 < 1) ↔ (∀ x : ℝ, x^2 ≥ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_square_lt_one_l1369_136923


namespace NUMINAMATH_CALUDE_divisor_calculation_l1369_136992

theorem divisor_calculation (dividend quotient remainder divisor : ℕ) : 
  dividend = 15968 ∧ quotient = 89 ∧ remainder = 37 ∧ 
  dividend = divisor * quotient + remainder → 
  divisor = 179 := by
  sorry

end NUMINAMATH_CALUDE_divisor_calculation_l1369_136992


namespace NUMINAMATH_CALUDE_marble_bowls_theorem_l1369_136954

theorem marble_bowls_theorem (capacity_ratio : Rat) (second_bowl_marbles : Nat) : 
  capacity_ratio = 3/4 → second_bowl_marbles = 600 →
  capacity_ratio * second_bowl_marbles + second_bowl_marbles = 1050 := by
  sorry

end NUMINAMATH_CALUDE_marble_bowls_theorem_l1369_136954


namespace NUMINAMATH_CALUDE_jose_investment_is_45000_l1369_136990

/-- Represents the investment and profit scenario of Tom and Jose --/
structure InvestmentScenario where
  tom_investment : ℕ
  jose_join_delay : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Jose's investment amount based on the given scenario --/
def calculate_jose_investment (scenario : InvestmentScenario) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that Jose's investment is 45000 given the specific scenario --/
theorem jose_investment_is_45000 :
  let scenario : InvestmentScenario := {
    tom_investment := 30000,
    jose_join_delay := 2,
    total_profit := 72000,
    jose_profit := 40000
  }
  calculate_jose_investment scenario = 45000 := by
  sorry

end NUMINAMATH_CALUDE_jose_investment_is_45000_l1369_136990


namespace NUMINAMATH_CALUDE_joan_total_cents_l1369_136935

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of quarters Joan has -/
def joan_quarters : ℕ := 6

/-- Theorem: Joan's total cents -/
theorem joan_total_cents : joan_quarters * quarter_value = 150 := by
  sorry

end NUMINAMATH_CALUDE_joan_total_cents_l1369_136935


namespace NUMINAMATH_CALUDE_intersection_implies_solution_l1369_136985

-- Define the linear function
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem intersection_implies_solution (k b : ℝ) :
  linear_function k b (-3) = 0 →
  (∃ x : ℝ, -k * x + b = 0) ∧
  (∀ x : ℝ, -k * x + b = 0 → x = 3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_solution_l1369_136985


namespace NUMINAMATH_CALUDE_square_difference_plus_two_cubed_l1369_136912

theorem square_difference_plus_two_cubed : (7^2 - 3^2 + 2)^3 = 74088 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_plus_two_cubed_l1369_136912


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1369_136932

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) → (a < -2 ∨ a > 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1369_136932


namespace NUMINAMATH_CALUDE_overlapping_squares_area_l1369_136993

/-- Given two overlapping squares with side length 12, where the overlap forms an equilateral triangle,
    prove that the area of the overlapping region is 108√3, and m + n = 111 -/
theorem overlapping_squares_area (side_length : ℝ) (m n : ℕ) :
  side_length = 12 →
  (m : ℝ) * Real.sqrt n = 108 * Real.sqrt 3 →
  n.Prime →
  m + n = 111 :=
by sorry

end NUMINAMATH_CALUDE_overlapping_squares_area_l1369_136993


namespace NUMINAMATH_CALUDE_perpendicular_slope_l1369_136922

/-- Given a line with equation 4x - 5y = 20, the slope of the perpendicular line is -5/4 -/
theorem perpendicular_slope (x y : ℝ) :
  (4 * x - 5 * y = 20) → (slope_of_perpendicular_line = -5/4) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l1369_136922


namespace NUMINAMATH_CALUDE_next_chime_together_l1369_136991

def town_hall_interval : ℕ := 18
def library_interval : ℕ := 24
def railway_interval : ℕ := 30

def minutes_in_hour : ℕ := 60

theorem next_chime_together (start_hour : ℕ) : 
  ∃ (hours : ℕ), 
    hours * minutes_in_hour = Nat.lcm town_hall_interval (Nat.lcm library_interval railway_interval) ∧ 
    hours = 6 := by
  sorry

end NUMINAMATH_CALUDE_next_chime_together_l1369_136991


namespace NUMINAMATH_CALUDE_equal_coins_after_transfer_l1369_136906

/-- Represents the amount of gold coins each merchant has -/
structure Merchants where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (m : Merchants) : Prop :=
  (m.ierema + 70 = m.yuliy) ∧ (m.foma - 40 = m.yuliy)

/-- The theorem to prove -/
theorem equal_coins_after_transfer (m : Merchants) 
  (h : satisfies_conditions m) : 
  m.foma - 55 = m.ierema + 55 := by
  sorry

#check equal_coins_after_transfer

end NUMINAMATH_CALUDE_equal_coins_after_transfer_l1369_136906


namespace NUMINAMATH_CALUDE_tree_watering_l1369_136926

theorem tree_watering (num_boys : ℕ) (trees_per_boy : ℕ) :
  num_boys = 9 →
  trees_per_boy = 3 →
  num_boys * trees_per_boy = 27 :=
by sorry

end NUMINAMATH_CALUDE_tree_watering_l1369_136926


namespace NUMINAMATH_CALUDE_equation_solution_l1369_136988

theorem equation_solution : ∃ x : ℝ, 300 * x + (12 + 4) * (1 / 8) = 602 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1369_136988


namespace NUMINAMATH_CALUDE_port_perry_wellington_ratio_l1369_136959

/-- The ratio of Port Perry's population to Wellington's population -/
def population_ratio (port_perry : ℕ) (wellington : ℕ) (lazy_harbor : ℕ) : ℚ :=
  port_perry / wellington

theorem port_perry_wellington_ratio :
  ∀ (port_perry wellington lazy_harbor : ℕ),
    wellington = 900 →
    port_perry = lazy_harbor + 800 →
    port_perry + lazy_harbor = 11800 →
    population_ratio port_perry wellington lazy_harbor = 7 := by
  sorry

#check port_perry_wellington_ratio

end NUMINAMATH_CALUDE_port_perry_wellington_ratio_l1369_136959


namespace NUMINAMATH_CALUDE_expression_simplification_l1369_136964

theorem expression_simplification (a : ℝ) (h : a = 2) : 
  (1 / (a + 1) - (a + 2) / (a^2 - 1) * (a^2 - 2*a + 1) / (a^2 + 4*a + 4)) * (a + 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1369_136964


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1369_136907

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x + y ≠ 8 → (x ≠ 2 ∨ y ≠ 6)) ∧
  (∃ x y : ℝ, (x ≠ 2 ∨ y ≠ 6) ∧ x + y = 8) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1369_136907


namespace NUMINAMATH_CALUDE_students_in_both_games_l1369_136965

theorem students_in_both_games (total : ℕ) (game_a : ℕ) (game_b : ℕ) 
  (h_total : total = 55) (h_game_a : game_a = 38) (h_game_b : game_b = 42) :
  ∃ x : ℕ, x = game_a + game_b - total ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_games_l1369_136965


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l1369_136952

theorem absolute_value_equation_product (x₁ x₂ : ℝ) : 
  (|4 * x₁| + 3 = 35) ∧ (|4 * x₂| + 3 = 35) ∧ (x₁ ≠ x₂) → x₁ * x₂ = -64 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l1369_136952
